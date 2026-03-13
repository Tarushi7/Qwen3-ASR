
#extracting files that are more cobtent less noise


import json
import numpy as np
from datasets import load_from_disk


rawdata = load_from_disk('/mnt/data/datasets/yt_sea_hf_v0.2/Emotional-YTB-SG_en_30_test')

with open("transcription_results.json", "r", encoding="utf-8") as f:
    transcriptions = json.load(f)

quality_data = []


for entry in transcriptions:
    idx = entry.get("index")
    
    audio_data = rawdata[idx]['context']['audio']
    waveform = np.array(audio_data['array'])
    sr = audio_data['sampling_rate']
    

    true_duration = len(waveform) / sr   #duration: Total Samples / Samples per Second
    if true_duration == 0:
        continue


    timestamps = entry.get('timestamps', [])
    text = entry.get('text', "")
    
    total_spoken_time = sum(ts['end'] - ts['start'] for ts in timestamps)
    speech_ratio = total_spoken_time / true_duration
     
    cpm = (len(text) / true_duration) * 60   #cpm
    

    if speech_ratio >= 0.70 and cpm >= 600:
        entry['true_duration'] = round(true_duration, 2)
        entry['speech_ratio'] = round(speech_ratio, 2)
        entry['cpm'] = round(cpm, 2)
        quality_data.append(entry)


with open("quality_data.json", "w", encoding="utf-8") as f:
    json.dump(quality_data, f, indent=4, ensure_ascii=False)

print(f"Filtered {len(quality_data)} files using exact durations.")


# 376 quality files out of 480
