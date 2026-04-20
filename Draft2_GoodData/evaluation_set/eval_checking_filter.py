import json
import os

# Paths to match your previous filtering script
FILTER_OUTPUT_DIR = "./filtered_transcriptions"
LANGUAGES = ["en_eval", "ms_eval", "zh_eval", "ta_eval"]

def generate_report():
    print("\n" + "="*65)
    print(f"{'Language':<15} | {'Total':<10} | {'Kept (✔)':<10} | {'Discard (✘)':<10} | {'Pass %':<8}")
    print("-" * 65)

    grand_total = 0
    grand_kept = 0
    grand_discard = 0

    for lang in LANGUAGES:
        kept_file = os.path.join(FILTER_OUTPUT_DIR, f"{lang}_neutral.json")
        discard_file = os.path.join(FILTER_OUTPUT_DIR, f"{lang}_discard.json")

        # Count kept items
        num_kept = 0
        if os.path.exists(kept_file):
            with open(kept_file, 'r') as f:
                num_kept = len(json.load(f))

        # Count discarded items
        num_discard = 0
        if os.path.exists(discard_file):
            with open(discard_file, 'r') as f:
                num_discard = len(json.load(f))

        total = num_kept + num_discard
        pass_rate = (num_kept / total * 100) if total > 0 else 0

        # Print row
        print(f"{lang:<15} | {total:<10} | {num_kept:<10} | {num_discard:<10} | {pass_rate:>6.1f}%")

        # Update totals for the bottom row
        grand_total += total
        grand_kept += num_kept
        grand_discard += num_discard

    print("-" * 65)
    total_pass_rate = (grand_kept / grand_total * 100) if grand_total > 0 else 0
    print(f"{'TOTAL':<15} | {grand_total:<10} | {grand_kept:<10} | {grand_discard:<10} | {total_pass_rate:>6.1f}%")
    print("="*65 + "\n")

if __name__ == "__main__":
    if not os.path.exists(FILTER_OUTPUT_DIR):
        print(f"Error: Directory '{FILTER_OUTPUT_DIR}' not found. Run the filter script first!")
    else:
        generate_report()