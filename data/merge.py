import json

records = []

for filepath in ["sroie_clean_text.jsonl", "batch_1_clean_text.jsonl"]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))



print(f"Total records: {len(records)}")



with open("merged_clean.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")