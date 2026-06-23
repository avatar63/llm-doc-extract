import json

INSTRUCTION = (
    "Extract the following fields from the OCR text as JSON: "
    "company_name, address, date, total_amount, line_items "
    "(each with item_name, quantity, price). "
    "Use null for any field that cannot be determined."
)

with open("xy_pairs.jsonl", "r") as infile, open("training_data.jsonl", "w") as outfile:
    
    for line in infile:
        record = json.loads(line)
        training_example = {
            "instruction": INSTRUCTION,
            "input": record["x"],
            "output": json.dumps(record["y"])
        }
        outfile.write(json.dumps(training_example) + "\n")

print("dataset is now in instruct tuned pairs")