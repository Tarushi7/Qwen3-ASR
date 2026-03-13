#much faster than thinking

import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. SETUP INSTRUCT MODEL
model_name = "Qwen/Qwen3-4B-Instruct-2507-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto", 
    device_map="auto"
)

def get_instruct_emotion(text, prompt_template):
    if not text.strip():
        return "neutral"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that labels emotional intensity in text."},
        {"role": "user", "content": prompt_template.format(text=text)}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    # Instruct models are concise, so we don't need many tokens
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=128 
    )
    
    # Extract only the newly generated tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    return content


# PICKING THE FILE
input_file = "en_dataset_neutral.json"
output_file = "en_dataset_instruct_results.json"

if os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # PROMPT
    test_prompt = "Does the following text sound emotionally intense or neutral? Answer with one word only. Text: '{text}'"

    print(f"Analysing {len(data)} files")

    for entry in data:
        raw_text = entry.get('text', "")
        # Get the Instruct model's label
        label = get_instruct_emotion(raw_text, test_prompt)
        
        # Add the new findings to the existing entry
        entry['instruct_label'] = label
        print(f"Index {entry['index']}: {label}")

    # SAVE RESULTS
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {output_file}")
else:
    print(f"Error: {input_file} not found.")

