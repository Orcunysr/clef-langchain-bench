"""Generative LLM baseline (Qwen3.5-9B) on the same 10 cases as bench_classification.py."""

import json
import re
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

# Reuse CASES from bench_classification.py without running its model code.
source = open("bench_classification.py", encoding="utf-8").read()
namespace = {}
exec(source[: source.index("def evaluate")], namespace)
CASES = namespace["CASES"]


def options(question):
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def build_prompt(case):
    lines = [f"STATE:\n{case['state'] if isinstance(case['state'], str) else json.dumps(case['state'], ensure_ascii=False)}",
             "", "QUESTIONS:"]
    for qid, q in case["questions"].items():
        lines.append(f"- {qid} ({q['type']}): {q.get('instructions', qid)}")
        if q["type"] == "choice":
            for k, v in q["criteria"].items():
                lines.append(f"    {k}: {v}")
        elif q["type"] == "score":
            for i, v in enumerate(q["criteria"]):
                lines.append(f"    {i}: {v}")
        else:
            lines.append("    true / false")
    keys = ", ".join(f'"{qid}": <one of {options(q)}>' for qid, q in case["questions"].items())
    lines += ["", f"Answer with only a JSON object: {{{keys}}}"]
    return "\n".join(lines)


SYSTEM = "You are a classifier. Read the state and answer every question with exactly one allowed option id."

path = snapshot_download("Qwen/Qwen3.5-9B")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(path)
model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
load_s = time.time() - t0
print(f"loaded in {load_s:.1f}s, GPU0 {torch.cuda.memory_allocated(0) / 2**30:.1f} GiB", flush=True)


def generate(case, thinking, max_new_tokens):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(case)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    seconds = time.time() - t
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens), inputs.input_ids.shape[1], seconds


def parse(answer):
    answer = answer.split("</think>")[-1]
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not match:
        return None
    try:
        return {k: str(v).lower() for k, v in json.loads(match.group(0)).items()}
    except json.JSONDecodeError:
        return None


results = []
generate(CASES[0], False, 64)  # warm-up
for mode, thinking, budget in (("no-think", False, 128), ("think", True, 4096)):
    for case in CASES:
        text, n_new, n_in, seconds = generate(case, thinking, budget)
        parsed = parse(text)
        for qid, gold in case["gold"].items():
            pred = (parsed or {}).get(qid)
            results.append({"model": f"qwen3.5-9b {mode}", "case": case["name"], "question": qid, "gold": gold,
                            "pred": pred, "correct": pred == gold, "s": round(seconds, 2),
                            "new_tokens": n_new, "input_tokens": n_in, "parsed": parsed is not None})
            print(f"[{mode}] {case['name']:<38} {qid:<12} gold={gold:<15} pred={str(pred):<15} "
                  f"{'OK' if pred == gold else 'XX'} {seconds:.2f}s {n_new}tok ({n_new / seconds:.1f} tok/s)", flush=True)
        if parsed is None:
            print(f"   !! unparsable output: {text[-300:]!r}", flush=True)

with open("llm_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)

print("\n=== SUMMARY ===")
for mode in ("no-think", "think"):
    rows = [r for r in results if r["model"] == f"qwen3.5-9b {mode}"]
    per_case = {r["case"]: (r["s"], r["new_tokens"]) for r in rows}
    secs = [s for s, _ in per_case.values()]
    toks = [n for _, n in per_case.values()]
    print(f"qwen3.5-9b {mode}: accuracy {sum(r['correct'] for r in rows)}/{len(rows)}, "
          f"mean {sum(secs) / len(secs):.2f} s/case (min {min(secs):.2f}, max {max(secs):.2f}), "
          f"mean {sum(toks) / len(toks):.0f} new tokens, {sum(toks) / sum(secs):.1f} tok/s, load {load_s:.0f} s")
