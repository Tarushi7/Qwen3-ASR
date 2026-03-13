import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model setup
model_name = "Qwen/Qwen3-4B-Thinking-2507-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

def get_llm_emotion(text, prompt_template):
    if not text.strip(): return "neutral", "empty"
    messages = [{"role": "user", "content": prompt_template.format(text=text)}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError: index = 0
    
    reasoning = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return answer, reasoning


# PICKING THE FILE
input_filename = "en_dataset_neutral.json"
output_filename = "en_dataset_with_LLM_labels.json"

with open(input_filename, "r") as f:
    data = json.load(f)

# PROMPT
prompt_A = "Label the emotional intensity of this text as 'High' or 'Neutral'. Text: {text}"

print(f"Analysing {len(data)} files")

for entry in data:
    text = entry.get('text', "")
    # Get the LLM's opinion
    label, logic = get_llm_emotion(text, prompt_A)
    
    # Add the new findings to the existing entry
    entry['llm_label'] = label
    entry['llm_reasoning'] = logic
    print(f"Processed Index {entry['index']}: LLM says {label}")

# SAVE RESULTS
with open(output_filename, "w") as f:
    json.dump(data, f, indent=4)

print(f"Experiment saved to {output_filename}")



