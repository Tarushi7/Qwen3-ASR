
#PART2: extracting emotional files from the high quality files
# result:

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

with open("/home/tarushi/tarushi_folder/Qwen3-ASR/TestDataTutorial/quality_data.json") as f:
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
best_data2 = []


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
            best_data2.append(batch_entries[idx])
            
    print(f"Processed batch from {i}. Current emotional count: {len(best_data2)}")

with open("best_data2.json", "w") as f:
    json.dump(best_data2, f, indent=4)

