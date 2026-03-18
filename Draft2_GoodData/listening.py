
import os
import soundfile as sf
from datasets import load_from_disk

# path = "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_30_dev"  #english
#path = "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_30_dev" #chinese
path = "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_30_dev" #malay
output_folder = 'extracted_wavs'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

ds = load_from_disk(path)

print(f"Starting conversion of {len(ds)} files...")

# #for all files:
# for i, entry in enumerate(ds):
#     audio_data = entry['context']['audio'] 
    
#     waveform = audio_data['array']
#     sample_rate = audio_data['sampling_rate']
    
#     file_path = os.path.join(output_folder, f"audio_{i}.wav")
    
#     sf.write(file_path, waveform, sample_rate)
#     if i % 100 == 0:
#         print(f"Converted {i} files...")
    

#for individual files:
target_index = 735
entry = ds[target_index]
audio = entry['context']['audio']

file_path = os.path.join(output_folder, f"check_ml_index_{target_index}.wav")
sf.write(file_path, audio['array'], audio['sampling_rate'])

print(f"Extracted index {target_index} to: {file_path}")


#EMOTION LABEL EXTRACTION (nested in meta_psudo)
meta = entry.get('other_attributes', {}).get('meta_psudo', {})
emotions = meta.get('emotion_2s', [])


print(f"Emotions detected (every 2s): {emotions}")




