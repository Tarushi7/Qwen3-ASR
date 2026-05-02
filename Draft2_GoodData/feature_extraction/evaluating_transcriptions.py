
import os
import json

# Define the directory where the filtered files are stored
FILTER_OUTPUT_DIR = "filtered_meralion_transcriptions"

# The 8 languages/lengths we processed
lang_configs = [
    {"lang": "en", "len": "30"}, {"lang": "zh", "len": "30"},
    {"lang": "ms", "len": "30"}, {"lang": "ta", "len": "30"},
    {"lang": "en", "len": "60"}, {"lang": "zh", "len": "60"},
    {"lang": "ms", "len": "60"}, {"lang": "ta", "len": "60"}
]

print(f"{'Dataset':<15} | {'Neutral':<10} | {'Discarded':<10} | {'Discard %':<10}")
print("-" * 55)

total_neutral = 0
total_discarded = 0

for info in lang_configs:
    lang = info['lang']
    length = info['len']
    dataset_id = f"{lang}_{length}"
    
    # Construct filenames based on your saving logic
    neutral_file = os.path.join(FILTER_OUTPUT_DIR, f"{dataset_id}_dataset_neutral.json")
    discard_file = os.path.join(FILTER_OUTPUT_DIR, f"{dataset_id}_dataset_discard.json")
    
    n_count = 0
    d_count = 0
    
    # Count Neutral
    if os.path.exists(neutral_file):
        with open(neutral_file, 'r') as f:
            n_count = len(json.load(f))
            
    # Count Discarded
    if os.path.exists(discard_file):
        with open(discard_file, 'r') as f:
            d_count = len(json.load(f))
            
    total_samples = n_count + d_count
    discard_pct = (d_count / total_samples * 100) if total_samples > 0 else 0
    
    total_neutral += n_count
    total_discarded += d_count
    
    print(f"{dataset_id:<15} | {n_count:<10} | {d_count:<10} | {discard_pct:>8.2f}%")

print("-" * 55)
overall_total = total_neutral + total_discarded
overall_pct = (total_discarded / overall_total * 100) if overall_total > 0 else 0
print(f"{'OVERALL':<15} | {total_neutral:<10} | {total_discarded:<10} | {overall_pct:>8.2f}%")

