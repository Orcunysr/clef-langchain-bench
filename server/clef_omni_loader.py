"""Multi-GPU + CPU offload loader for Cloudflare/clef-omni.

The release ``load_release_model`` pins the whole backbone to one device
(``device_map={"": device}``), which needs ~64 GB of VRAM. This loader spreads
the thinker over both GPUs and CPU RAM with ``device_map="auto"`` and moves the
joint schema head's inputs to the head's device.
"""

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file

DEFAULT_MAX_MEMORY = {0: "20GiB", 1: "20GiB", "cpu": "100GiB"}


def _patch_head_devices(jsm):
    """Make JointSchemaHead.forward move its tensor inputs to the head's device."""
    if getattr(jsm.JointSchemaHead.forward, "_device_patched", False):
        return
    original = jsm.JointSchemaHead.forward

    def forward(self, hidden_states, input_ids, attention_mask, records, output_embedding_weight):
        device = self.hidden_norm.weight.device
        return original(
            self,
            hidden_states.to(device),
            input_ids.to(device),
            attention_mask.to(device),
            records,
            output_embedding_weight.to(device),
        )

    forward._device_patched = True
    jsm.JointSchemaHead.forward = forward


def _read_tensor(path: Path, key: str) -> torch.Tensor:
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    with safe_open(str(path / index["weight_map"][key]), framework="pt") as handle:
        return handle.get_tensor(key)


def load_omni(
    model_path="Cloudflare/clef-omni",
    max_memory=None,
    head_device="cuda:1",
    dtype=torch.bfloat16,
):
    from transformers import AutoConfig, AutoProcessor, Qwen3OmniMoeForConditionalGeneration
    from transformers.utils import logging as transformers_logging

    path = Path(model_path)
    if not path.is_dir():
        path = Path(snapshot_download(str(model_path)))
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import joint_schema_model as jsm

    _patch_head_devices(jsm)

    config = AutoConfig.from_pretrained(path)
    config.enable_audio_output = False  # skip talker / code2wav
    verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        backbone, loading_info = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            path,
            config=config,
            dtype=dtype,
            device_map="auto",
            max_memory=max_memory or DEFAULT_MAX_MEMORY,
            output_loading_info=True,
        )
    finally:
        transformers_logging.set_verbosity(verbosity)
    unexpected = [k for k in loading_info.get("unexpected_keys") or () if not k.startswith(("talker.", "code2wav."))]
    missing = list(loading_info.get("missing_keys") or ())
    if missing or unexpected:
        raise RuntimeError(f"weights did not load cleanly: missing={missing[:10]} unexpected={unexpected[:10]}")

    device_map = getattr(backbone, "hf_device_map", {})
    placement = {}
    for module, device in device_map.items():
        placement.setdefault(str(device), []).append(module)
    for device, modules in placement.items():
        print(f"[loader] {device}: {len(modules)} modules (e.g. {modules[:3]})", flush=True)

    thinker = backbone.thinker
    thinker.config.use_cache = False

    head_config = json.loads((path / "joint_head_config.json").read_text(encoding="utf-8"))
    head = jsm.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    head = head.to(device=head_device, dtype=dtype)

    model = jsm.ClefModel(thinker, head).eval()

    # ClefModel keeps lm_head only for its weight (lexical option embeddings).
    # If accelerate offloaded it, the parameter sits on "meta"; load a real copy.
    weight = model.output_embeddings.weight
    if weight.device.type in ("meta", "cpu"):
        real = _read_tensor(path, "thinker.lm_head.weight").to(device=head_device, dtype=dtype)
        lm_head = torch.nn.Linear(real.shape[1], real.shape[0], bias=False, device="meta")
        lm_head.weight = torch.nn.Parameter(real, requires_grad=False)
        model.output_embeddings = lm_head
        print(f"[loader] lm_head was on {weight.device}; reloaded onto {head_device}", flush=True)

    processor = AutoProcessor.from_pretrained(path)
    return model, processor, jsm
