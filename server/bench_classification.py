import gc
import json
import sys
import time

import torch
from huggingface_hub import snapshot_download

CASES = [
    {
        "name": "01 support routing",
        "state": "I was charged $49 twice this month for the Pro plan. Please refund the duplicate.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                           "criteria": {"billing": "Payments, refunds, invoices", "technical": "Bugs, outages, errors",
                                        "sales": "New purchases, upgrades, quotes", "account": "Login, password, profile"}},
        },
        "gold": {"department": "billing"},
    },
    {
        "name": "02 sentiment (sarcasm)",
        "state": "Oh great, the app crashed again right before I saved. Fantastic work, really.",
        "questions": {
            "sentiment": {"type": "choice", "instructions": "What is the sentiment of the review?",
                          "criteria": {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}},
            "sarcastic": {"type": "noul", "instructions": "Is the text sarcastic?"},
        },
        "gold": {"sentiment": "negative", "sarcastic": "true"},
    },
    {
        "name": "03 phishing email",
        "state": {"from": "security@paypa1-support.com", "subject": "Account suspended",
                  "body": "We detected unusual activity. Verify your identity within 24 hours at http://paypa1-verify.xyz/login or your account will be closed."},
        "questions": {
            "label": {"type": "choice", "instructions": "Classify this email.",
                      "criteria": {"phishing": "Credential theft or impersonation", "spam": "Unsolicited marketing",
                                   "legitimate": "Genuine message"}},
        },
        "gold": {"label": "phishing"},
    },
    {
        "name": "04 legit email (false-positive check)",
        "state": {"from": "no-reply@github.com", "subject": "[GitHub] A new SSH key was added to your account",
                  "body": "The following SSH key was added to your account: MacBook. If you believe this key was added in error, you can remove it at https://github.com/settings/keys"},
        "questions": {
            "label": {"type": "choice", "instructions": "Classify this email.",
                      "criteria": {"phishing": "Credential theft or impersonation", "spam": "Unsolicited marketing",
                                   "legitimate": "Genuine message"}},
        },
        "gold": {"label": "legitimate"},
    },
    {
        "name": "05 Turkish complaint",
        "state": "Siparişim 10 gündür kargoda bekliyor, müşteri hizmetleri de telefonu açmıyor. Paramı geri istiyorum!",
        "questions": {
            "topic": {"type": "choice", "instructions": "What is the main topic?",
                      "criteria": {"shipping": "Delivery or shipping delay", "product_quality": "Defective or poor product",
                                   "pricing": "Price complaint", "praise": "Compliment"}},
            "wants_refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
            "urgency": {"type": "score", "criteria": ["Low", "Medium", "High"]},
        },
        "gold": {"topic": "shipping", "wants_refund": "true", "urgency": "2"},
    },
    {
        "name": "06 news topic",
        "state": "The central bank raised interest rates by 50 basis points on Thursday, citing persistent inflation in services and housing.",
        "questions": {
            "topic": {"type": "choice", "instructions": "What is the news category?",
                      "criteria": {"economy": "Economy and finance", "sports": "Sports", "politics": "Politics and elections",
                                   "technology": "Technology", "health": "Health"}},
        },
        "gold": {"topic": "economy"},
    },
    {
        "name": "07 intent (banking77-like)",
        "state": "My card got stuck in the ATM at the train station and it won't give it back.",
        "questions": {
            "intent": {"type": "choice", "instructions": "What is the user's intent?",
                       "criteria": {"card_swallowed": "ATM kept the card", "lost_or_stolen_card": "Card lost or stolen",
                                    "declined_card_payment": "Card payment declined", "cash_withdrawal_charge": "Fee on ATM withdrawal",
                                    "activate_card": "Activate a new card"}},
        },
        "gold": {"intent": "card_swallowed"},
    },
    {
        "name": "08 toxicity",
        "state": "You absolute idiot, nobody wants your garbage opinions here. Get lost.",
        "questions": {
            "toxic": {"type": "noul", "instructions": "Is the comment toxic or abusive?"},
            "severity": {"type": "score", "criteria": ["None", "Mild", "Moderate", "Severe threat"]},
        },
        "gold": {"toxic": "true", "severity": "2"},
    },
    {
        "name": "09 NLI",
        "state": {"premise": "All flights from Istanbul to Berlin were cancelled on Monday due to a storm.",
                  "hypothesis": "A flight from Istanbul to Berlin departed on Monday."},
        "questions": {
            "relation": {"type": "choice", "instructions": "Does the premise entail the hypothesis?",
                         "criteria": {"entailment": "Hypothesis follows", "neutral": "Cannot tell",
                                      "contradiction": "Hypothesis is false given premise"}},
        },
        "gold": {"relation": "contradiction"},
    },
    {
        "name": "10 log severity",
        "state": "2026-10-11T03:14:07Z db-primary ERROR: could not write to file \"pg_wal/0000000100000A2B\": No space left on device",
        "questions": {
            "severity": {"type": "choice", "instructions": "What is the operational severity?",
                         "criteria": {"info": "Informational", "warning": "Degraded, no impact yet",
                                      "critical": "Service impacting, needs immediate action"}},
            "component": {"type": "choice", "instructions": "Which component is affected?",
                          "criteria": {"database": "Database", "network": "Network", "frontend": "Web frontend",
                                       "auth": "Authentication"}},
        },
        "gold": {"severity": "critical", "component": "database"},
    },
]


def evaluate(name, model, processor, jsm, device):
    rows = []
    for case in CASES:
        record = {"state": case["state"], "questions": case["questions"]}
        encoded = jsm.encode_record(processor.tokenizer, record, processor=processor)
        batch = jsm.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device(device))
        torch.cuda.synchronize()
        t = time.time()
        with torch.inference_mode():
            logits = model(batch)[0]
        torch.cuda.synchronize()
        ms = 1000 * (time.time() - t)
        for question, question_logits in zip(encoded.questions, logits):
            probs = dict(zip(question.option_ids, question_logits.float().softmax(-1).tolist()))
            pred = max(probs, key=probs.get)
            gold = case["gold"][question.question_id]
            rows.append({"model": name, "case": case["name"], "question": question.question_id, "gold": gold,
                         "pred": pred, "correct": pred == gold, "p_gold": round(probs[gold], 4),
                         "p_pred": round(probs[pred], 4), "ms": round(ms)})
            print(f"[{name}] {case['name']:<38} {question.question_id:<12} gold={gold:<15} pred={pred:<15} "
                  f"p_gold={probs[gold]:.3f} {'OK' if pred == gold else 'XX'} {ms:.0f}ms", flush=True)
    return rows


results = []
load_s = {}

# --- flash (single GPU) ---
flash_path = snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, flash_path)
import joint_schema_model as flash_jsm  # noqa: E402

t0 = time.time()
model, processor = flash_jsm.load_release_model(flash_path, device="cuda:0")
load_s["flash"] = time.time() - t0
evaluate("flash-warmup", model, processor, flash_jsm, "cuda:0")[:0]  # warm-up pass, discarded
results += evaluate("flash", model, processor, flash_jsm, "cuda:0")
del model, processor
gc.collect()
torch.cuda.empty_cache()
sys.path.remove(flash_path)
del sys.modules["joint_schema_model"]

# --- omni (2 GPUs + CPU offload) ---
from clef_omni_loader import load_omni  # noqa: E402

t0 = time.time()
model, processor, omni_jsm = load_omni()
load_s["omni"] = time.time() - t0
evaluate("omni-warmup", model, processor, omni_jsm, "cuda:0")[:0]
results += evaluate("omni", model, processor, omni_jsm, "cuda:0")

with open("compare_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)

print("\n=== SUMMARY ===")
for name in ("flash", "omni"):
    rows = [r for r in results if r["model"] == name]
    acc = sum(r["correct"] for r in rows) / len(rows)
    mean_p = sum(r["p_gold"] for r in rows) / len(rows)
    case_ms = {r["case"]: r["ms"] for r in rows}
    print(f"{name}: accuracy {acc:.1%} ({sum(r['correct'] for r in rows)}/{len(rows)}), "
          f"mean p_gold {mean_p:.3f}, mean latency {sum(case_ms.values()) / len(case_ms):.0f} ms/case (min {min(case_ms.values())}, max {max(case_ms.values())}), load {load_s[name]:.0f} s")
