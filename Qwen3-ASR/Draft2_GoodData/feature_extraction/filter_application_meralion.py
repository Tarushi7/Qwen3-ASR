
# since meralion has no timestamps like the qwen, guess the talk time based on word count
# 2.3 words per second based on avg speech: 130-150 WPM
# cap the talk time so it's never longer than the actual audio

import os
import json
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
from text_based_filtering import get_text_metrics, get_stutter_stats, get_redundancy_score, get_final_tier

# Define the output directory
FILTER_OUTPUT_DIR = "filtered_meralion_transcriptions"
os.makedirs(FILTER_OUTPUT_DIR, exist_ok=True)

# unique keys to prevent overwriting
lang_configs = {
    "en_30": {"lang": "en", "len": "30", "path": "Emotional-YTB-MY_en_30_dev", "stutter": 0.40},
    "zh_30": {"lang": "zh", "len": "30", "path": "Emotional-YTB-MY_zh_30_dev", "stutter": 0.50},
    "ms_30": {"lang": "ms", "len": "30", "path": "Emotional-YTB-MY_ms_30_dev", "stutter": 0.40},
    "ta_30": {"lang": "ta", "len": "30", "path": "Emotional-YTB-MY_ta_30_dev", "stutter": 0.40},
    "en_60": {"lang": "en", "len": "60", "path": "Emotional-YTB-MY_en_60_dev", "stutter": 0.40},
    "zh_60": {"lang": "zh", "len": "60", "path": "Emotional-YTB-MY_zh_60_dev", "stutter": 0.50},
    "ms_60": {"lang": "ms", "len": "60", "path": "Emotional-YTB-MY_ms_60_dev", "stutter": 0.40},
    "ta_60": {"lang": "ta", "len": "60", "path": "Emotional-YTB-MY_ta_60_dev", "stutter": 0.40}
}

for config_key, info in lang_configs.items():
    print(f"\nProcessing: {config_key.upper()}")
    
    dataset_full_path = f"/home/q-wang/data/hf_egs/{info['path']}"
    trans_file_path = f"meralion_outputs/{config_key}_transcriptions.json"

    if not os.path.exists(dataset_full_path) or not os.path.exists(trans_file_path):
        print(f"Skipping {config_key}: File or Dataset not found.")
        continue

    ds = load_from_disk(dataset_full_path)
    
    with open(trans_file_path, "r", encoding="utf-8") as f:
        trans_data = json.load(f)
        trans_lookup = {item['idx']: item for item in trans_data}

    results = {"neutral": [], "discard": []}

    for i in tqdm(range(len(ds)), desc=f"Filtering {config_key}"):
        item = ds[i]
        trans_info = trans_lookup.get(i, {})
        
        current_text = trans_info.get('transcription', "")
        audio_data = item.get('context', {}).get('audio', {})
        y = np.array(audio_data.get('array') or [])
        sr = audio_data.get('sampling_rate') or 16000
        true_duration = len(y) / sr 
        lang_code = info['lang'] # Extract lang code for the functions

        # 1. Clean the text
        clean_text = current_text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
        
        # 2. Build Synthetic Timestamps (Logic updated to use Language-Specific Tokenization)
        # We call get_text_metrics twice: Once with empty timestamps to get the word list,
        # then again with the synthetic timestamps to get the final metrics.
        
        # Initial pass to get the correct 'words' list for ZH/TA/EN
        initial_info = get_text_metrics(clean_text, [], y, sr, lang_code)
        words = initial_info['words']
        word_count = initial_info['word_count']
        
        current_timestamps = []
        talk_time = 0
        
        if word_count > 0:
            # Estimate: 2.3 tokens per second
            estimated_talk_time = word_count / 2.3  
            talk_time = min(estimated_talk_time, true_duration)
            
            word_dur = talk_time / max(1, word_count)
            start_offset = (true_duration - talk_time) / 2
            current_timestamps = [[start_offset + (j * word_dur), start_offset + ((j + 1) * word_dur)] for j in range(word_count)]

        # 3. Final Feature Extraction (Now with timestamps and lang_code)
        text_info = get_text_metrics(text=clean_text, timestamps=current_timestamps, y=y, sr=sr, lang_code=lang_code)
        
        # 4. Final Override for talk_time
        if text_info.get('talk_time', 0) == 0 and word_count > 0:
            text_info['talk_time'] = talk_time
            text_info['speech_ratio'] = talk_time / true_duration

        # 5. Secondary Metrics
        stutter_info = get_stutter_stats(text_info['words'])
        redundancy_val = get_redundancy_score(text_info['words']) 

        # 6. Final Decision Logic (Passed lang_code)
        comp_tier = get_final_tier(
            text_info, 
            stutter_info, 
            redundancy_score=redundancy_val, 
            lang_code=lang_code, # Added
            stutter_threshold=info["stutter"]
        )

        final_tier = "discard" if "Discard" in comp_tier else "neutral"

        # 7. Package Results
        clean_text_features = {k: v for k, v in text_info.items() if k != 'words'}
        master_dict = {
            "index": i,
            "audio_id": trans_info.get("audio_id", f"sample_{i}"),
            "text": current_text,
            "label": final_tier,
            "text_features": {
                **clean_text_features, 
                "word_repetition": stutter_info.get('word_repetition', 0.0), # Updated key
                "redundancy_score": redundancy_val 
            }
        }
        
        results[final_tier].append(master_dict)

    # 8. Save Results
    print(f"Summary for {config_key}:")
    for tier_name, data_list in results.items():
        out_filename = f"{info['lang']}_{info['len']}_dataset_{tier_name}.json"
        out_full_path = os.path.join(FILTER_OUTPUT_DIR, out_filename)
        
        with open(out_full_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        
        print(f" - Saved {len(data_list)} items to {out_full_path}")

print("\nAll datasets filtered and saved to 'filtered_meralion_transcriptions'.") 

