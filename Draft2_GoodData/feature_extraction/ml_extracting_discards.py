
import json
import os
import re

# 1. SETUP
input_file = "ml_dataset_instruct_analysis.json"
output_file = "ml_discards_only.json"

def filter_discards():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    discards_only = []

    print(f"{'INDEX':<10} | {'VERDICT':<10}")
    print("-" * 25)

    for entry in full_data:
        # Get the model's response string
        results = entry.get('instruct_model_results', {})
        full_res = results.get('full_response', "")
        
        # Use regex to find "Verdict": "DISCARD" (case-insensitive)
        # This ensures we don't accidentally grab the prompt text
        if re.search(r'"Verdict":\s*"DISCARD"', full_res, re.IGNORECASE):
            discards_only.append(entry)
            print(f"{entry.get('index', 'N/A'):<10} | DISCARD")

    # 2. SAVE THE NEW FILE
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(discards_only, f, indent=4, ensure_ascii=False)

    print("-" * 25)
    print(f"Created: {output_file}")
    print(f"Total Discards Found: {len(discards_only)}")

if __name__ == "__main__":
    filter_discards()