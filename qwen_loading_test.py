
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav", 
    language=None, # set "English" to force the language
)

print(results[0].language)
print(results[0].text)


# trying to use it on the processed batch file:
my_audio_file = '/home/tarushi/tarushi_folder/processed_batch_1413_keC7RIg3bDM_41.wav'
print(f"File: {my_audio_file}")

results = model.transcribe(
    audio=my_audio_file,
    language=None, # It will auto-detect, or set to "en" for English
)

print(f"Language: {results[0].language}")
print(f"Transcription: {results[0].text}")

