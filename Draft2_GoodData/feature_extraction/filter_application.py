
import os
import json
from datasets import load_from_disk
import numpy as np

from text_based_filtering import get_text_metrics, get_stutter_stats, get_redundancy_score, get_final_tier


#saving all paths as a dictionary
lang_configs = {
    "en": {
        "dataset_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_30_dev",
        "trans_file": "meralion_outputs/en_30.json",
        "stutter_threshold": 0.40
    },
    "cn": {
        "dataset_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_30_dev",
        "trans_file": "meralion_outputs/h_30.json",
        "stutter_threshold": 0.50
    },
    "ml": {
        "dataset_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_30_dev",
        "trans_file": "meralion_outputs/ml_30.json",
        "stutter_threshold": 0.40
    },
    "ta": {
        "dataset_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_30_dev",
        "trans_file": "meralion_outputs/ta_30.json",
        "stutter_threshold": 0.40
    },
        "en": {
        "dataset_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_60_dev",
        "trans_file": "meralion_outputs/en_60.json",
        "stutter_threshold": 0.40
    },
    "cn": {
        "dataset_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_60_dev",
        "trans_file": "meralion_outputs/cn_60.json",
        "stutter_threshold": 0.50
    },
    "ml": {
        "dataset_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_60_dev",
        "trans_file": "meralion_outputs/ml_60.json",
        "stutter_threshold": 0.40
    },
    "ta": {
        "dataset_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_60_dev",
        "trans_file": "meralion_outputs/ta_60.json",
        "stutter_threshold": 0.40
    }
}



for lang_code, config in lang_configs.items():
    print(f"\nProcessing Language: {lang_code.upper()}")
    
    # Load Dataset and Transcriptions logic stays the same...
    if not os.path.exists(config["dataset_path"]): continue
    ds = load_from_disk(config["dataset_path"])
    with open(config["trans_file"], "r") as f:
        trans_lookup = {item['index']: item for item in json.load(f)}

    results = {"neutral": [], "discard": []}

    for i in range(len(ds)):
        item = ds[i]
        trans_info = trans_lookup.get(i, {})
        current_text = trans_info.get('text', "")
        current_timestamps = trans_info.get('timestamps', [])

        audio_data = item.get('context', {}).get('audio', {})
        y = np.array(audio_data.get('array') or [])
        sr = audio_data.get('sampling_rate') or 16000

        text_info = get_text_metrics(text=current_text, timestamps=current_timestamps, y=y, sr=sr)
        
        stutter_info = get_stutter_stats(text_info['words'])
        redundancy_val = get_redundancy_score(text_info['words']) 


        comp_tier = get_final_tier(
            text_info, 
            stutter_info, 
            redundancy_score=redundancy_val, 
            stutter_threshold=config["stutter_threshold"]
        )

        if "Discard" in comp_tier:
            final_tier = "discard"
        else:
            final_tier = "neutral"


        clean_text_features = {k: v for k, v in text_info.items() if k != 'words'}
        
        master_dict = {
            "index": i,
            "text": current_text,
            "label": final_tier,
            "text_features": {
                **clean_text_features, 
                "word_repetition": stutter_info['stutter_score'],
                "redundancy_score": redundancy_val 
            }
        }
        
        results[final_tier].append(master_dict)

    
    # SAVE RESULTS WITH LANGUAGE PREFIX:
    print(f"Summary for {lang_code.upper()}:")
    for tier_name, data_list in results.items():
            filename = f"{lang_code}_60_dataset_{tier_name}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=4)
            print(f"Saved {len(data_list)} files to {filename}")


print("All languages processed successfully.")


