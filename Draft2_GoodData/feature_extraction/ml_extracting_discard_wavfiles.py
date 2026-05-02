import json
import os
import soundfile as sf
from datasets import load_from_disk

# 1. SETUP
json_input = "ml_discards_only.json"
dataset_path = "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_30_dev"
export_dir = "evaluation_directory_ml"

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# 2. LOAD DATA
with open(json_input, "r") as f:
    discards = json.load(f)

ds = load_from_disk(dataset_path)

print(f"Extracting {len(discards)} audio clips from dataset...")

# 3. EXTRACT AND SAVE
for entry in discards:
    idx = entry['index']
    
    # Grab the audio data from the Hugging Face dataset
    item = ds[idx]
    audio_data = item['context']['audio']['array']
    sampling_rate = item['context']['audio']['sampling_rate']
    
    # Save it as a physical .wav file in your export folder
    file_name = f"index_{idx}.wav"
    output_path = os.path.join(export_dir, file_name)
    
    sf.write(output_path, audio_data, sampling_rate)

# 4. INCLUDE THE TEXT (JSON)
with open(os.path.join(export_dir, "metadata.json"), "w") as f:
    json.dump(discards, f, indent=4)

# 5. ZIP IT UP
import shutil
shutil.make_archive("evaluation_folder_ml", 'zip', export_dir)

print(f" Created evaluation_folder_ml.zip")