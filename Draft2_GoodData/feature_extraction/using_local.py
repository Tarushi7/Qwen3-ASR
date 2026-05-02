import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. CRITICAL: This prevents the 'NotImplementedError' you saw earlier
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

model_name = "Qwen/Qwen2.5-3B-Instruct" # Using 3B or 4B is safer for Mac RAM

print("Loading model onto Mac GPU (Metal)...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # Mac GPU loves float16
    device_map={"": "mps"}     # This forces it onto the Metal chip
)

print("Model loaded on MPS!")

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

def get_detailed_instruct_emotion(text, prompt_template):
    messages = [
        {"role": "system", "content": "You are a linguistic expert specializing in emotional prosody analysis."},
        {"role": "user", "content": prompt_template.format(text=text)}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    
    # max_new_tokens=512 is the sweet spot for reasoning + labels
    outputs = model.generate(**inputs, max_new_tokens=1024)
    output_ids = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

# 3. FILE PROCESSING
input_file = "en_dataset_neutral.json"
output_file = "en_dataset_instruct_analysis.json"
test_limit = 100

if os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # Slice the data AND the texts to process immediately
    data = full_data[:test_limit]
    texts_to_process = [entry.get('text', "") for entry in data]

    print(f"Starting analysis on {len(data)} files...")

    for i, text in enumerate(texts_to_process):
        # Safety check: skip if ASR failed to produce text
        if not text.strip():
            data[i]['instruct_model_results'] = {"full_response": "N/A - Empty Text"}
            continue

        # Run model
        response = get_detailed_instruct_emotion(text, detailed_prompt)
        
        # Save results back to our 'data' slice
        data[i]['instruct_model_results'] = {
            "full_response": response,
            "model_used": model_name
        }
        
        # Proactive Progress Tracker
        print(f"[{i+1}/{test_limit}] Processed Index: {data[i]['index']}")

    # 4. FINAL SAVE
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {output_file}")

else:
    print(f"Error: {input_file} not found. Check your file path!")

