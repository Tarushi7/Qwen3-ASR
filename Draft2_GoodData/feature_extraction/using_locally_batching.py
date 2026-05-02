
import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. MAC-SPECIFIC CONFIG
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("mps")

model_name = "Qwen/Qwen2.5-3B-Instruct" 
batch_size = 10 

print(f"Loading {model_name} onto Mac GPU (Metal)...")

# Load Tokenizer with Left Padding
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map={"": "mps"}
).eval() # .eval() is better for local inference

print(" Model loaded on MPS!")



# Prompt
detailed_prompt = """

PROMPT 4:
  
### INSTRUCTION:
You are a high-precision linguistic filter for a paralinguistic dataset.
CRITICAL NOTE: Transcriptions may contain sudden, disjointed shifts in emotion or topic. A file might contain 90percent filler and 10percent intense emotion. If AT LEAST ONE phrase shows significant emotional weight (Anger, Panic, Joy, Distress, Fear, Surprise, Confusion, Disgust, Empathy, etc.), the file must be marked as KEEP. We are strictly filtering for emotional "spikes" regardless of the surrounding neutral context.
 
### RULES:
1. Look at each phrase (roughly every 4-6 words).
2. Assign an emotion label to EACH segment as a list. Use 'Neutral' for flat text and specific labels for emotions.
3. Identify the specific phrase (Evidence) that justifies the highest emotional intensity.
4. Final Verdict: KEEP if you sense any paralinguistic spike in one or more of the phrases. Otherwise, DISCARD.
5. Only decide KEEP or DISCARD after looking at emotion labels
 
### EXAMPLES:
Text: "I'll see you at five. Oh my god! Is that a snake? Get back!"
Emotion_Label: ["Neutral", "Neutral", "Surprise", "Fearful", "Fearful", "Panic"],
Verdict: "KEEP",
Evidence: "Oh my god! Is that a snake? Get back!",
Reasoning: "Text starts as a neutral appointment but shifts abruptly to life-threatening panic. The presence of 'Panic' labels requires a KEEP verdict."
 
Text: "The data shows a slight increase in margin. We should review this next Tuesday."
Emotion_Label: ["Neutral", "Neutral", "Neutral", "Neutral", "Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "Entire transcription is professional, informational, and lacks any emotional phrases."
 
Text: "I'm so happy for you! Anyway, I need to go buy some milk and eggs now."
Emotion_Label: ["Happy", "Happy", "Happy", "Neutral", "Neutral", "Neutral", "Neutral"],
Verdict: "KEEP",
Evidence: "I'm so happy for you!",
Reasoning: "Despite the second half being mundane filler, the initial segment contains high-intensity Joy, earning the "KEEP" status"
 
Text: "If you love someone, hold on to them. Tomorrow you might not have the chance. Sometimes you go on long trips to science centers."
Emotion_Label: ["Neutral", "Caring", "Neutral", "Anxiety", "Fear", "Neutral", "Neutral", "Neutral"],
Verdict: "KEEP",
Evidence: "Tomorrow you might not have the chance.",
Reasoning: "The mention of potential loss introduces a spike of anxiety/fear which is keep-worthy, even though the text ends with unrelated neutral content."
 
Text: "I see a chair, I see a table. Tuck the chair into the table"
Emotion_Label: ["Neutral", "Neutral", "Calmness"],
Verdict: "DISCARD",
Evidence: "None.",
Reasoning: "Entire transcription is solely descriptive and gives an order."
 
### TASK:
Text: '{text}'
 
### OUTPUT FORMAT:
Respond ONLY with a valid JSON object containing the keys: "Emotion_Label", "Verdict", "Evidence", and "Reasoning".

"""

def process_batch(batch_entries):
    #Prepare messages for the whole batch
    formatted_prompts = []
    for entry in batch_entries:
        text = entry.get('text', "")
        messages = [
            {"role": "system", "content": "You are a linguistic expert specializing in emotional prosody analysis."},
            {"role": "user", "content": detailed_prompt.format(text=text)}
        ]
        # Apply Qwen's specific chat format
        formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Tokenize and pad the batch
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False, # Consistent results
            temperature=0.1
        )

    # Decode results
    batch_responses = []
    for i in range(len(batch_entries)):
        # Only take the newly generated tokens
        output_ids = outputs[i][len(inputs.input_ids[i]):]
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        batch_responses.append(decoded)
    
    return batch_responses

# FILE PROCESSING 
input_file = "en_dataset_neutral.json"
output_file = "en_dataset_instruct_analysis.json"
test_limit = 50

if os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    data = full_data[:test_limit]
    print(f"Starting Local Metal analysis in batches of {batch_size}...")

    # Process in chunks
    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        
        # Filter out empty texts in the chunk
        valid_chunk = [e for e in chunk if e.get('text', "").strip()]
        
        if valid_chunk:
            responses = process_batch(valid_chunk)
            
            # Map back to data
            for idx, response in enumerate(responses):
                valid_chunk[idx]['instruct_model_results'] = {
                    "full_response": response,
                    "model_used": model_name
                }
        
        print(f"Processed {min(i + batch_size, len(data))}/{len(data)} files...")

    # SAVE
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"✨ Success! Results saved to {output_file}")

else:
    print(f"Error: {input_file} not found!")