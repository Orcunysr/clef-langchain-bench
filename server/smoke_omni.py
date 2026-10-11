import json
import subprocess
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

from clef_omni_loader import load_omni

HERE = Path(__file__).resolve().parent


def make_speech_wav(text: str, out: Path) -> Path:
    """Synthesize speech with Windows SAPI, then resample to 16 kHz mono with av."""
    raw = out.with_suffix(".raw.wav")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SetOutputToWaveFile('{raw}');$s.Speak('{text}');$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    samples = []
    with av.open(str(raw)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            samples.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        samples.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    pcm = np.concatenate(samples).astype(np.int16)
    with av.open(str(out), "w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=16000, layout="mono")
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 16000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    raw.unlink(missing_ok=True)
    print(f"  wav: {out} ({len(pcm) / 16000:.1f}s @16kHz mono)")
    return out


t0 = time.time()
model, processor, jsm = load_omni()
print(f"loaded in {time.time() - t0:.1f}s")
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i} alloc {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")


def run(record):
    encoded = jsm.encode_record(processor.tokenizer, record, processor=processor)
    batch = jsm.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda:0"))
    t = time.time()
    with torch.inference_mode():
        logits = model(batch)[0]
    torch.cuda.synchronize()
    print(f"  ({len(encoded.input_ids)} tokens, {time.time() - t:.1f} s)")
    for question, question_logits in zip(encoded.questions, logits):
        probs = question_logits.float().softmax(-1).tolist()
        print("  ", question.question_id, {k: round(v, 4) for k, v in zip(question.option_ids, probs)})


SUPPORT_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
    },
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"},
}

print("\n== 1. Text: support message")
run({"state": "Our checkout started returning errors and orders are blocked.", "questions": SUPPORT_QUESTIONS})

print("\n== 2. SystemOne: invoice question (billing, not urgent)")
print(json.dumps(jsm.systemone(model, processor, {
    "model": "clef-omni",
    "state": "Hi, could you send me a copy of last month's invoice when you get a chance? No rush.",
    "questions": SUPPORT_QUESTIONS,
})["answers"], indent=2))

print("\n== 3. Image (synthetic red circle + blue square + text)")
image = Image.new("RGB", (448, 448), "white")
draw = ImageDraw.Draw(image)
draw.ellipse((40, 40, 220, 220), fill="red")
draw.rectangle((260, 260, 420, 420), fill="blue")
draw.text((60, 330), "TOTAL: 42.00 USD", fill="black")
run({
    "state": {"task": "Describe the attached image."},
    "images": [image],
    "questions": {
        "circle_color": {
            "type": "choice",
            "instructions": "What color is the circle?",
            "criteria": {"red": "Red", "green": "Green", "blue": "Blue"},
        },
        "has_square": {"type": "noul", "instructions": "Is there a square in the image?"},
    },
})

print("\n== 4. Audio: spoken support request (state has no text about the issue)")
wav = make_speech_wav(
    "Hello, this is urgent. Our checkout page is down and customers cannot place orders. Please help right now.",
    HERE / "support_16k.wav",
)
run({
    "state": {"task": "A customer left the attached voicemail. Route it."},
    "audio": [str(wav)],
    "questions": SUPPORT_QUESTIONS,
})

print("\n== 5. Audio: calm billing voicemail")
wav2 = make_speech_wav(
    "Hi, I think I was charged twice on my last invoice. Whenever you have time next week, could you check it? Thanks.",
    HERE / "billing_16k.wav",
)
run({
    "state": {"task": "A customer left the attached voicemail. Route it."},
    "audio": [str(wav2)],
    "questions": SUPPORT_QUESTIONS,
})
