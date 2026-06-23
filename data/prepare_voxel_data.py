from glob import glob
import json
import pandas as pd

FILE_PATHS = glob("/home/grizzly/.cache/kagglehub/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr/versions/3/batch_1/batch_1/*.csv")

output_records = []

def extract_text_from_batch_1(csv_path):

    for ocr_data in pd.read_csv(csv_path)["OCRed Text"]:
        output_records.append({
            "source":"batch_1",
            "clean_text":ocr_data})


for path in FILE_PATHS:
    extract_text_from_batch_1(path)


print(f"Extracted {len(output_records)} SROIE records")



with open("batch_1_clean_text.jsonl", "w") as f:
    for r in output_records:
        f.write(json.dumps(r) + "\n")