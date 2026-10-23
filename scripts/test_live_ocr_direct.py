import requests
import json
from pathlib import Path

img_path = Path("data/raw/VedOCR/VedOCR_kaggle_release/sample_pages/00000002.png")
assert img_path.exists(), f"Image not found at {img_path}"

with open(img_path, "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/api/ocr/direct",
        files={"file": ("00000002.png", f, "image/png")},
        data={"lang": "san+hin", "preprocessing": "auto", "mode": "multi-pass"}
    )

print("HTTP Status Code:", r.status_code)
res = r.json()
print("Extraction status:", res.get("status"))
print("Lines extracted:", res.get("lines_count"))
print("Characters extracted:", res.get("character_count"))
print("Is multi-pass:", res.get("is_multi_pass"))
print("Mean confidence:", res.get("mean_confidence"))
print("Processing time ms:", res.get("processing_time_ms"))
print("Provenance breakdown:", json.dumps(res.get("provenance_breakdown"), indent=2))
print("Uncertain lines count:", res.get("uncertain_lines_count"))

print("\n--- First 5 Reconciled Manuscript Lines with Provenance ---")
for line in res.get("lines_detail", [])[:5]:
    print(f"Line {line['line_num']}: [{line['pass_name']}] (Conf: {line['mean_confidence']}%, Uncertain: {line.get('is_uncertain')})")
    safe_text = line['text'].encode('ascii', errors='backslashreplace').decode('ascii')
    print(f"  Text: {safe_text}")

assert res.get("status") == "SUCCESS"
assert res.get("is_multi_pass") is True
assert res.get("lines_count", 0) > 0
print("\n[SUCCESS] Live Full-Page Manuscript Multi-Pass Extraction Verified!")
