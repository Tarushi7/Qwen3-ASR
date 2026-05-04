
import numpy as np
import json
import tempfile
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel
from datasets import load_from_disk


model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    max_inference_batch_size=32 # This tells the internal engine how many to handle
)

data = load_from_disk('/mnt/data/datasets/yt_sea_hf_v0.2/Emotional-YTB-SG_en_30_test')

# 2. Batch Processing 
batch_size = 10
all_results = []
all_transcriptions = []

for i in range(0, len(data), batch_size):
    batch_items = data.select(range(i, min(i + batch_size, len(data))))
    batch_audio_paths = []
    temp_files = []

    # temp files for each batch
    for item in batch_items:
        audio_node = item['context']['audio']
        waveform = np.array(audio_node['array'])
        sr = audio_node['sampling_rate']
        
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
        sf.write(tmp.name, waveform, sr)
        batch_audio_paths.append(tmp.name)
        temp_files.append(tmp) # keep reference

    print(f"Transcribing Batch: {i} to {i + len(batch_items)}:")
    
    # GPU Parallel processing by passing a list of paths, the model triggers its internal batching
    results = model.transcribe(
        audio=batch_audio_paths, 
        language="English",
        return_time_stamps=True
    )

    for idx, r in enumerate(results):
        # We create a dictionary for each file so it's easy to read later
        entry = {
            "index": i + idx,
            "text": r.text,
            "language": r.language,
            "timestamps": [
                {"start": ts.start_time, "end": ts.end_time, "text": ts.text} 
                for ts in r.time_stamps
            ] if r.time_stamps else []
        }
        all_transcriptions.append(entry)
    
    all_results.extend(results)

    # Clean up temp files for this batch
    for f in temp_files:
        f.close() 

output_filename = "TestDataTutorial/transcription_results.json"

with open(output_filename, "w", encoding="utf-8") as f:
    # indent=4 makes the file "pretty" and readable for humans
    # ensure_ascii=False keeps special characters looking right
    json.dump(all_transcriptions, f, indent=4, ensure_ascii=False)

print(f"All results saved to {output_filename} ---")

