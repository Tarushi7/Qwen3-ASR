
#using qwen3-asr on the mounted test data

#to exclude the flash attention
import os
os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"
os.environ["FLASH_ATTENTION_SKIP_CUDA_CHECK"] = "1"

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from datasets import load_from_disk
import soundfile as sf
import tempfile



model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B", #this line is for the timestamps [remove if no need]
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)


path = '/mnt/data/datasets/yt_sea_hf_v0.2/Emotional-YTB-SG_en_30_test'
data = load_from_disk(path)

print(data.column_names)
#data[0]
print(data[0]['context'].keys())
#print(f"Total indexes: {len(data)}")

for i in range(len(data)):
    audio_node = data[i]['context']['audio']
    #Convert the 'list' from the arrow file into a numpy array:
    audio_array = np.array(audio_node['array']) #this was the one causing the "FileNotFound" error because of the .arrow format
    sr = audio_node['sampling_rate']
    
    # Create the temporary file bridge
    # We use delete=True so it cleans itself up immediately after the 'with' block
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio_array, sr)
        
        print(f" Transcribing Index {i}:")
        results = model.transcribe(
            audio=tmp.name, 
            language="English", #note that it is "English" and not "en"
            return_time_stamps=True #[for timestamps only: remove if no need]
        )
        print(f"Output {i}: {results}")

#Methods that do not work:
#1.converting the values into a tensor 
#2.convering into 
#3.defining a path using a class function like below:
# class NestedAudioDataset:
#     def __init__(self, arrow_path):
#         self.dataset = load_from_disk(arrow_path)

#     def __len__(self):
#         return len(self.dataset)

#     def get_audio_path(self, index):
#         item = self.dataset[index]
#         audio_node = item['context']['audio'] 
        
#         return audio_node['path']

# data_nest = NestedAudioDataset('/mnt/data/datasets/yt_sea_hf_v0.2/Emotional-YTB-SG_en_30_test')

# for i in range(len(data_nest)):
#     path = data_nest.get_audio_path(i)
#     results = model.transcribe(audio=path)


