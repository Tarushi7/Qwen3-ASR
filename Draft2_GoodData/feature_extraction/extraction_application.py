
import os
import json

from datasets import load_from_disk
import librosa
import numpy as np

from sound_based_extraction import get_loudness_features, get_pitch_features, get_rhythm_features, get_final_tier_s
from text_based_filtering import get_text_metrics, get_repetition_stats, get_final_tier

path = "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_30_dev"
ds = load_from_disk(path)

# all_rates = [item['sampling_rate'] for item in ds['context']['audio']]
# unique_rates = np.unique(all_rates)
# print(f"Unique Sampling Rates: {unique_rates}")
# print(f"Total files: {len(all_rates)}")

# ds[0].keys()
# ds[0]['context'].keys() # context keys
# ds[0]['context']['audio'].keys() # audio keys


#loading the transcription as a dictionary
with open("transcriptions.json", "r") as f:
    trans_data = json.load(f)

trans_lookup = {item['index']: item for item in trans_data}



results = {
    "highly_emotional": [],
    "neutral": [],
    "discard": []
}

test_limit = 30

for i in range(test_limit):
    item = ds[i]

    context_data = item.get('context', {})

    # force 'None' to become an empty string
    current_text = context_data.get('text') or "" 

    current_timestamps = context_data.get('timestamps')

    trans_info = trans_lookup.get(i, {})
    current_text = trans_info.get('text', "")
    current_timestamps = trans_info.get('timestamps', [])

    # Audio extraction
    audio_data = context_data.get('audio', {})
    wf = audio_data.get('array') or []
    sr = audio_data.get('sampling_rate') or 16000

    y = np.array(wf)


    # -- sound-based extractions
    loud_stats = get_loudness_features(y)
    pitch_stats = get_pitch_features(y, sr)
    rhythm_stats = get_rhythm_features(y, sr)
    
    # -- computational extractions
    text_info = get_text_metrics(text = current_text, timestamps = current_timestamps, y = y, sr=sr)
    repetition_info = get_repetition_stats(text = current_text)


    # Sound-Based: Emotion data
    sound_tier = get_final_tier_s(pitch_stats, loud_stats, rhythm_stats, text_info)
    # Computational: Higer Quality Data
    comp_tier = get_final_tier(text_info, repetition_info)


    #-- DECISION RULE
 
    if comp_tier == "Low Quality: Discard":
        final_tier = "discard"
    elif sound_tier == "High Intensity":
        final_tier = "highly_emotional"
    else:
        # Low Intensity and Neutral files both will be captured here
        final_tier = "neutral"


    #MASTER DICTIONARY
    master_dict = {
        "index": i,
        "text": current_text,
        "label": final_tier,
        "sound_features": {**loud_stats, **pitch_stats, **rhythm_stats},
        "text_features": {**text_info, **repetition_info}
    }
    results[final_tier].append(master_dict)



# -- display summary stats
total_files = len(ds)
print("FINAL DATASET SUMMARY:")

for tier_name, data_list in results.items():
    filename = f"dataset_{tier_name}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4)
    print(f"Saved {len(data_list)} files to {filename}")





