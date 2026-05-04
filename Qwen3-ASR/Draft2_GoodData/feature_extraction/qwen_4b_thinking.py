import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# SETUP
model_name = "Qwen/Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# detailed_prompt = """
# Analyze the emotional intensity of the following speech transcription: '{text}'. 

# 1. Rate the intensity on a scale of 1 to 10 (1 = Monotone/Neutral, 10 = Extreme/Panic/Rage).
# 2. Identify the primary emotion.
# 3. Explain your reasoning based on word choice and linguistic cues.

# Format your final answer exactly like this:
# Intensity: [Score]
# Emotion: [Type]
# Reasoning: [Explanation]
# """

detailed_prompt = """
### INSTRUCTION:
Analyze the emotional intensity of the text below. 
Note that the text may contain multiple speakers or different unrelated sections. 

You must respond ONLY in the following format:
Intensity: [1-10] (Rate based on the peak/highest emotional moment found in any section)
Emotion: [Category]
Reasoning: [Identify if there is a shift in topic or emotion and explain the highest point]
Sections: [Identify the number of sections and which section has the highest point]

### EXAMPLE:
Text: "The weather is nice today. WAIT! STOP THE CAR! THERE IS A DOG IN THE ROAD!"
Intensity: 9
Emotion: Panic/Alarm
Reasoning: Text shifts from neutral observation to an urgent crisis. Peak intensity is in the second half.

### TASK:
Text: '{text}'
"""


def get_detailed_thinking_emotion(text, prompt_template):
    messages = [{"role": "user", "content": prompt_template.format(text=text)}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    # Thinking models need room to "thought-dump"
    outputs = model.generate(**inputs, max_new_tokens=1024) 
    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

    try:
        index = output_ids.index(151668) # End of thinking block
    except ValueError: 
        index = 0

    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return answer, thinking

# DATA LOADING & SLICING (Do this ONLY ONCE)
input_file = "en_dataset_neutral.json"
output_file = "en_dataset_thinking_analysis.json"

with open(input_file, "r", encoding="utf-8") as f:
    full_data = json.load(f)

test_limit = 100 
data = full_data[:test_limit] # Slice to 100
texts_to_process = [entry.get('text', "") for entry in data]

print(f"Starting Test Run: {len(data)} files out of {len(full_data)}")

# SINGLE PROCESSING LOOP
for i, current_text in enumerate(texts_to_process):
    if not current_text.strip():
        data[i]['thinking_results'] = {"answer": "N/A", "thought_process": "Empty text"}
        continue

    # Get both the Reasoning and the Internal Logic
    answer, thinking = get_detailed_thinking_emotion(current_text, detailed_prompt)
    
    # Save everything into the current entry
    data[i]['thinking_results'] = {
        "analysis": answer, 
        "internal_thought": thinking
    }

    print(f"[{i+1}/{test_limit}] Done Index {data[i]['index']}")

# FINAL SAVE
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"Experiment complete! Results saved to {output_file}")


