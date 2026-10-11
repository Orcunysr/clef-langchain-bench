"""Local inference server for the Windows box.

GPU0: Cloudflare/clef-flash  -> POST /v1/systemone        (Jev/SystemOne compatible)
GPU1: Qwen/Qwen3.5-9B        -> POST /v1/chat/completions (OpenAI compatible, tool calls)

Listens on 127.0.0.1 only; reach it from the Mac through an SSH tunnel.
"""

import json
import re
import sys
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

DECISION_DEVICE = "cuda:0"
LLM_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
CHAT_MODELS = {"qwen-fast": False, "qwen-think": True}  # model name -> enable_thinking
DEFAULT_MAX_TOKENS = {"qwen-fast": 1024, "qwen-think": 1024}
ANSWER_TOKENS_AFTER_FORCED_STOP = 512

print("loading clef-flash ...", flush=True)
clef_path = snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, clef_path)
import joint_schema_model as jsm  # noqa: E402

clef_model, clef_processor = jsm.load_release_model(clef_path, device=DECISION_DEVICE)

print("loading Qwen3.5-9B ...", flush=True)
qwen_path = snapshot_download("Qwen/Qwen3.5-9B")
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path)
qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    qwen_path, dtype=torch.bfloat16, device_map={"": LLM_DEVICE}
).eval()

clef_lock = threading.Lock()
qwen_lock = threading.Lock()
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "decision": "clef-flash", "chat": list(CHAT_MODELS)}


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    body.setdefault("model", "clef-flash")
    started = time.time()
    try:
        with clef_lock:
            response = jsm.systemone(clef_model, clef_processor, body)
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    summary = {
        qid: answer.get("choice", answer.get("noul", answer.get("score")))
        for qid, answer in response["answers"].items()
    }
    print(f"[systemone] {summary} {1000 * (time.time() - started):.0f} ms", flush=True)
    return response


# ---------------------------------------------------------------- chat completions

TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _to_template_messages(messages):
    """OpenAI messages -> what the Qwen chat template expects (tool-call args as dicts)."""
    converted = []
    for message in messages:
        message = dict(message)
        if message.get("content") is None:
            message["content"] = ""
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"] or "{}")
        converted.append(message)
    return converted


def _parameter_types(tools):
    types = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        properties = (function.get("parameters") or {}).get("properties") or {}
        types[function["name"]] = {name: spec.get("type") for name, spec in properties.items()}
    return types


def _parse_tool_calls(text, tools):
    types = _parameter_types(tools)
    calls = []
    for name, body in TOOL_CALL.findall(text):
        arguments = {}
        for key, raw in PARAMETER.findall(body):
            if types.get(name, {}).get(key) == "string":
                arguments[key] = raw
            else:
                try:
                    arguments[key] = json.loads(raw)
                except json.JSONDecodeError:
                    arguments[key] = raw
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        })
    return calls, TOOL_CALL.sub("", text).strip()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model", "qwen-fast")
    if model_name not in CHAT_MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model {model_name!r}; use one of {list(CHAT_MODELS)}")
    if body.get("stream"):
        raise HTTPException(status_code=400, detail="streaming is not supported")
    tools = body.get("tools") or None
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS[model_name]

    prompt = qwen_tokenizer.apply_chat_template(
        _to_template_messages(body["messages"]),
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=CHAT_MODELS[model_name],
    )
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(LLM_DEVICE)
    prompt_length = inputs.input_ids.shape[1]
    started = time.time()
    forced = False
    with qwen_lock, torch.inference_mode():
        output = qwen_model.generate(**inputs, max_new_tokens=int(max_tokens), do_sample=False)
        text = qwen_tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        if CHAT_MODELS[model_name] and "</think>" not in text:
            # Reasoning ran out of budget: close it and let the model answer.
            forced = True
            closing = qwen_tokenizer(
                "\n\nI have to stop thinking and answer now.\n</think>\n\n",
                return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(LLM_DEVICE)
            continued = torch.cat([output, closing], dim=1)
            output = qwen_model.generate(
                input_ids=continued, attention_mask=torch.ones_like(continued),
                max_new_tokens=ANSWER_TOKENS_AFTER_FORCED_STOP, do_sample=False,
            )
            text = qwen_tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    new_tokens = output.shape[1] - prompt_length

    reasoning = None
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "").strip()
    tool_calls, content = _parse_tool_calls(text.strip(), tools)
    finished = "tool_calls" if tool_calls else "stop"
    print(f"[chat] {model_name} {new_tokens} tok {time.time() - started:.1f}s forced_stop={forced} "
          f"tools={[c['function']['name'] for c in tool_calls]} finish={finished}", flush=True)

    message = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finished}],
        "usage": {
            "prompt_tokens": prompt_length,
            "completion_tokens": new_tokens,
            "total_tokens": prompt_length + new_tokens,
        },
    }


if __name__ == "__main__":
    print("ready on http://127.0.0.1:30000", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=30000, log_level="warning")
