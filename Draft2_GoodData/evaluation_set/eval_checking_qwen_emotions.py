import json
import os
import re
from collections import Counter

# --- CONFIGURATION ---
RESULTS_DIR = "qwen_instruct_results"
analysis_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_qwen_analysis.json")]

def extract_labels_from_response(raw_response):
    """Extracts the Emotion_Label list from the LLM JSON response."""
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0).replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            labels = parsed.get("Emotion_Label", [])
            # Convert to a list of lower-case strings if it's a list
            if isinstance(labels, list):
                return [str(l).lower() for l in labels]
            return [str(labels).lower()]
    except:
        pass
    return []

print("\n" + "="*80)
print(f"{'Folder Emotion':<20} | {'Total':<8} | {'Most Common LLM Labels'}")
print("-" * 80)

for filename in analysis_files:
    file_path = os.path.join(RESULTS_DIR, filename)
    print(f"\nDATASET: {filename.upper()}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dictionary to store labels found for each folder type
    # Key: folder name (e.g. "angry"), Value: List of all LLM labels found
    stats = {}

    for entry in data:
        # Get the folder ground truth (e.g., "angry_occ4_wav" -> "angry")
        folder_raw = entry.get("emotion_subfolder", "unknown")
        folder_clean = folder_raw.split('_')[0] # Grabs 'angry' from 'angry_occ4_wav'
        
        results = entry.get("instruct_model_results", {})
        raw_response = results.get("full_response", "")
        
        llm_labels = extract_labels_from_response(raw_response)
        
        if folder_clean not in stats:
            stats[folder_clean] = []
        
        stats[folder_clean].extend(llm_labels)

    # Print results for this language
    for folder, labels in sorted(stats.items()):
        total_files_in_folder = len([e for e in data if e.get("emotion_subfolder", "").startswith(folder)])
        
        # Filter out 'neutral' to see the "hits"
        hits = [l for l in labels if l != "neutral"]
        common = Counter(hits).most_common(3)
        common_str = ", ".join([f"{k} ({v})" for k, v in common])
        
        print(f"{folder:<20} | {total_files_in_folder:<8} | {common_str}")

print("="*80)