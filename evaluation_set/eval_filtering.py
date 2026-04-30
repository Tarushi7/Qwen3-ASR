from collections import Counter
import string
import re
import os
import json
import librosa
import numpy as np
from tqdm import tqdm


def get_text_metrics(text, timestamps, y, sr, lang_code):
    true_duration = len(y) / sr
    total_spoken_time = sum(ts[1] - ts[0] for ts in timestamps)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()

    if lang_code == "zh":
        tokens = re.findall(r'[\u4e00-\u9fff]', text)
    elif lang_code == "ta":
        tokens = re.findall(r'[\u0b80-\u0bff]+|[a-zA-Z0-9]+', text)
    else:
        tokens = re.findall(r'[a-zA-Z0-9]+', text)

    words = [t.lower() for t in tokens]
    word_count = len(words)
    
    safe_dur = max(true_duration, 0.001)
    wpm = (word_count / safe_dur) * 60
    
    return {
        "true_duration": true_duration,
        "talk_time": total_spoken_time,
        "speech_ratio": total_spoken_time / safe_dur,
        "wpm": wpm,
        "word_count": word_count,
        "words": words 
    } 

def get_stutter_stats(words, window_size=5):
    if not words or len(words) < window_size:
        return {"word_repetition": 0, "total_words": len(words) if words else 0}
    total_words = len(words)
    stutter_windows = 0
    for i in range(len(words) - window_size + 1):
        window = words[i : i + window_size]
        window_counts = Counter(window)
        if any(count >= 3 for count in window_counts.values()):
            stutter_windows += 1
    return {"word_repetition": round(stutter_windows / (total_words - window_size + 1), 4), "total_words": total_words}

def get_redundancy_score(words, n=3):
    if not words or len(words) < n:
        return 0.0
    grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return round((len(grams) - len(set(grams))) / len(grams), 4)

def get_final_tier(text_info, stutter_info, redundancy_score, lang_code, stutter_threshold):
    min_wpm = {"en": 70, "ms": 70, "zh": 110, "ta": 50}.get(lang_code, 70)
    if text_info['wpm'] < min_wpm: return "Low Quality: Discard"
    if text_info['speech_ratio'] < 0.50: return "Low Quality: Discard"
    if stutter_info['word_repetition'] > stutter_threshold: return "Low Quality: Discard"
    if redundancy_score > 0.4: return "Low Quality: Discard"
    return "High Quality"



TRANSCRIPT_DIR = "./meralion_outputs"
FILTER_OUTPUT_DIR = "./filtered_transcriptions"
os.makedirs(FILTER_OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000

lang_configs = {
    "ta_eval": {"lang": "ta", "stutter": 0.40},
    "ms_eval": {"lang": "ms", "stutter": 0.40},
    "en_eval": {"lang": "en", "stutter": 0.40},
    "zh_eval": {"lang": "zh", "stutter": 0.50}
}

for config_key, info in lang_configs.items():
    trans_file_path = os.path.join(TRANSCRIPT_DIR, f"{config_key}_transcriptions.json")
    
    if not os.path.exists(trans_file_path):
        print(f"Skipping {config_key}: Transcription file not found.")
        continue

    print(f"\nProcessing: {config_key.upper()}")
    
    with open(trans_file_path, "r", encoding="utf-8") as f:
        trans_data = json.load(f)

    results = {"neutral": [], "discard": []}

    for item in tqdm(trans_data, desc=f"Filtering {config_key}"):
        if item.get("error") or not item.get("transcription"):
            results["discard"].append({**item, "label": "discard", "reason": "Model Transcription Error"})
            continue

        audio_path = item.get("source_audio_path")
        current_text = item.get("transcription", "")
        lang_code = info['lang']

        try:
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            true_duration = len(y) / sr
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            continue

        clean_text = current_text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
        
        # initial pass for word count
        initial_info = get_text_metrics(clean_text, [], y, sr, lang_code)
        words = initial_info['words']
        word_count = initial_info['word_count']
        
        estimated_talk_time = word_count / 2.3  
        talk_time = min(estimated_talk_time, true_duration)
        
        # dummy timestamps for the talk_time ratio check
        dummy_ts = [[0, talk_time]] if talk_time > 0 else []

        text_info = get_text_metrics(text=clean_text, timestamps=dummy_ts, y=y, sr=sr, lang_code=lang_code)
        stutter_info = get_stutter_stats(text_info['words'])
        redundancy_val = get_redundancy_score(text_info['words']) 

        comp_tier = get_final_tier(
            text_info, 
            stutter_info, 
            redundancy_score=redundancy_val, 
            lang_code=lang_code,
            stutter_threshold=info["stutter"]
        )

        final_tier = "discard" if "Discard" in comp_tier else "neutral"

        # removing the meotion form the ID
        raw_audio_id = item.get("audio_id", "")
        subfolder = item.get("emotion_subfolder", "")
 
        prefix_to_remove = subfolder.replace("_wav", "") + "_"
        clean_audio_id = raw_audio_id.replace(prefix_to_remove, "")
        
        clean_text_features = {k: v for k, v in text_info.items() if k != 'words'}
        master_dict = {
            "index": item.get("idx"),
            "audio_id": clean_audio_id, 
            "emotion_subfolder": subfolder,
            "text": current_text,
            "label": final_tier,
            "source_audio_path": audio_path,
            "text_features": {
                **clean_text_features, 
                "word_repetition": stutter_info.get('word_repetition', 0.0), 
                "redundancy_score": redundancy_val 
            }
        }
        
        results[final_tier].append(master_dict)

    # Saving as language-specific results
    for tier_name, data_list in results.items():
        out_filename = f"{config_key}_{tier_name}.json"
        out_full_path = os.path.join(FILTER_OUTPUT_DIR, out_filename)
        
        with open(out_full_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        
    print(f"Summary for {config_key}: {len(results['neutral'])} kept, {len(results['discard'])} discarded.")

print(f"\nAll datasets filtered and saved to '{FILTER_OUTPUT_DIR}'.")

