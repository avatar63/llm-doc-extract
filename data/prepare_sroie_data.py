import json
from glob import glob

FILE_PATHS = glob("/home/grizzly/.cache/kagglehub/datasets/hariwh0/sroie-scanned-invoice/versions/1/SROIE_invoice/train/box/*.txt")



def extract_text_from_sroie(file_path):
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 9:
                text = ','.join(parts[8:])
                lines.append(text)
    return '\n'.join(lines)


output_records = []

for filepath in FILE_PATHS:

    clean_text = extract_text_from_sroie(filepath)
    if clean_text.strip():
        output_records.append({
            "source": "sroie",
            "clean_text": clean_text
        })

print(f"Extracted {len(output_records)} SROIE records")

# Save for Claude normalization step
with open("sroie_clean_text.jsonl", "w") as f:
    for r in output_records:
        f.write(json.dumps(r) + "\n")