#using qwen LLM (Qwen2.5-7B-Instruct) to extract the transcriptions that are emotional 

import gc
gc.collect()

import torch
torch.cuda.empty_cache()

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

with open("/home/tarushi/tarushi_folder/Qwen3-ASR/TestDataTutorial/transcription_results.json") as f:
    results = json.load(f)

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left"


quant_config = BitsAndBytesConfig(load_in_4bit=True)

llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

batch_size = 8
emotion_data = []

# for entry in results:
#     transcripts = entry['text']
#     prompt = f"From the data analyse which audios at least 1 emotional speakers. Respond with only 'emotional' and 'not_emotional'. Transcript: {transcripts}"
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     outputs = llm_model.generate(**inputs, max_new_tokens=7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     if "YES" in response.upper():
#         emotion_data.append(entry)


for i in range(0, len(results), batch_size):
    batch_entries = results[i : i + batch_size]
    
    prompts = [
        f"Analyze if the speaker is emotional. Respond with only 'TRUE' or 'FALSE'. Transcript: {entry['text']}"
        for entry in batch_entries
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    outputs = llm_model.generate(**inputs, max_new_tokens=10)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for idx, full_response in enumerate(responses):
        answer = full_response.split("Transcript:")[-1].upper() #EXTRACTING ONLY THE T/F
        
        if "TRUE" in answer:
            emotion_data.append(batch_entries[idx])
            
    print(f"Processed batch from {i}. Current emotional count: {len(emotion_data)}")

with open("emotion_data.json", "w") as f:
    json.dump(emotion_data, f, indent=4)


# 262 emotional files out of 480
