import json
import os
import shutil
import re
from tqdm import tqdm

ANALYSIS_FILE = "qwen_instruct_results/ms_eval_qwen_analysis.json"
EXPORT_DIR = "malay_discards_package"
AUDIO_EXPORT_DIR = os.path.join(EXPORT_DIR, "wav_files")

os.makedirs(AUDIO_EXPORT_DIR, exist_ok=True)

def get_verdict(raw_response):
    if not raw_response: return "UNKNOWN"
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0).replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            return str(parsed.get("Verdict", "")).upper()
    except:
        pass
    upper_raw = raw_response.upper()
    if '"VERDICT": "DISCARD"' in upper_raw or '"VERDICT":"DISCARD"' in upper_raw:
        return "DISCARD"
    return "KEEP"

with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

discarded_samples = []
for entry in data:
    results = entry.get("instruct_model_results", {})
    raw_response = results.get("full_response", "")
    
    if "DISCARD" in get_verdict(raw_response):
        discarded_samples.append(entry)

audit_batch = discarded_samples[:100]
print(f"Found {len(discarded_samples)} total discards. Extracting 100 for audit.")

audit_metadata = []

for entry in tqdm(audit_batch, desc="Copying files"):
    
    src_path = entry.get("source_audio_path")
    audio_id = entry.get("audio_id")
    
    if src_path and os.path.exists(src_path):
       
        dest_filename = f"idx{entry['index']}_{audio_id}.wav"
        dest_path = os.path.join(AUDIO_EXPORT_DIR, dest_filename)
        
        shutil.copy2(src_path, dest_path)
        
        entry['local_audit_path'] = dest_filename
        audit_metadata.append(entry)
    else:
        print(f"Warning: Could not find audio at {src_path}")

metadata_path = os.path.join(EXPORT_DIR, "ms_eval_discards.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(audit_metadata, f, indent=4, ensure_ascii=False)

print(f"\nCOMPLETE")
print(f"Audio files are in: {AUDIO_EXPORT_DIR}")
print(f"Metadata is in: {metadata_path}")
