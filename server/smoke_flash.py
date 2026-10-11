import json
import sys
import time

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw

path = snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, path)
from joint_schema_model import collate_records, encode_record, load_release_model, systemone  # noqa: E402

t0 = time.time()
model, processor = load_release_model(path, device="cuda:0")
print(f"loaded in {time.time() - t0:.1f}s, GPU0 alloc {torch.cuda.memory_allocated(0) / 2**30:.1f} GiB", flush=True)


def run(record):
    encoded = encode_record(processor.tokenizer, record, processor=processor)
    batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda:0"))
    t = time.time()
    with torch.inference_mode():
        logits = model(batch)[0]
    torch.cuda.synchronize()
    print(f"  ({len(encoded.input_ids)} tokens, {1000 * (time.time() - t):.0f} ms)")
    for question, question_logits in zip(encoded.questions, logits):
        probs = question_logits.float().softmax(-1).tolist()
        print("  ", question.question_id, {k: round(v, 4) for k, v in zip(question.option_ids, probs)})


print("\n== 1. Model card example (invoice)")
run({
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "What is the invoice status?",
            "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
        },
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    },
})

print("\n== 2. SystemOne support message (department / urgency / outage)")
response = systemone(model, processor, {
    "model": "clef-flash",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
})
print(json.dumps(response, indent=2, ensure_ascii=False))

print("\n== 3. Image example (synthetic: red circle + text on white)")
image = Image.new("RGB", (448, 448), "white")
draw = ImageDraw.Draw(image)
draw.ellipse((40, 40, 220, 220), fill="red")
draw.rectangle((260, 260, 420, 420), fill="blue")
draw.text((60, 330), "TOTAL: 42.00 USD", fill="black")
image.save("synthetic.png")
run({
    "state": {"task": "Describe the attached image."},
    "images": [image],
    "questions": {
        "shapes": {
            "type": "choice",
            "instructions": "Which shapes are visible?",
            "criteria": {
                "circle_and_square": "A circle and a square",
                "only_circle": "Only a circle",
                "triangle": "A triangle",
            },
        },
        "circle_color": {
            "type": "choice",
            "instructions": "What color is the circle?",
            "criteria": {"red": "Red", "green": "Green", "blue": "Blue"},
        },
        "has_text": {"type": "noul", "instructions": "Does the image contain readable text?"},
        "is_photo": {"type": "noul", "instructions": "Is this a real-world photograph?"},
    },
})
