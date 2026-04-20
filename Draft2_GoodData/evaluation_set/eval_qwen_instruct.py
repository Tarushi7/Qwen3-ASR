import json

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map={"": 0},        
    attn_implementation="sdpa" 
)


INPUT_DIR = "filtered_transcriptions"
OUTPUT_DIR = "qwen_instruct_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


neutral_files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_eval_neutral.json")]

detailed_prompt = """

### INSTRUCTION:
You are a high-precision linguistic filter for a paralinguistic dataset.
CRITICAL NOTE: Transcriptions may contain sudden, disjointed shifts in emotion or topic. A file might contain 90 percent filler and 10 percent intense emotion. If AT LEAST ONE phrase shows significant emotional weight (Anger, Panic, Joy, Distress, Fear, Surprise, Confusion, Disgust, Empathy, etc.), the file must be marked as KEEP. We are strictly filtering for emotional phrases regardless of the surrounding neutral context.

 
### RULES:
1. Look at each phrase (roughly every 4-6 words).
2. Provide a chronological list of the primary emotions detected in the text. List the unique emotional states in order and Use 'Neutral' for flat text and specific labels for emotions.(e.g., ["Neutral", "Anger", "Happy", "Neutral", "Annoyed"]).
3. Identify the specific phrase (Evidence) that justifies the highest emotional intensity.
4. Final Verdict: KEEP if you sense any paralinguistic spike in one or more of the phrases. Otherwise, DISCARD.
5. Make sure EVERY response has a 'Verdict'
6. If a specific emotion label is consistent throughout the entire file (e.g., the speaker is crying or angry from start to finish), you must also mark it as KEEP. Do not discard just because there is no variation.
7. Only decide KEEP or DISCARD after looking at emotion labels
 

### MULTILINGUAL EXAMPLES:
--- English ---
Text: "I'll see you at five. Oh my god! Is that a snake? Get back!"
Emotion_Label: ["Neutral", "Surprise", "Fearful", "Panic"],
Verdict: "KEEP",
Evidence: "Oh my god! Is that a snake? Get back!",
Reasoning: "Text starts as a neutral appointment but shifts abruptly to life-threatening panic. The presence of 'Panic' labels requires a KEEP verdict."
 
Text: "The data shows a slight increase in margin. We should review this next Tuesday."
Emotion_Label: ["Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "Entire transcription is professional, informational, and lacks any emotional phrases."
 
Text: "I'm so happy for you! Anyway, I need to go buy some milk and eggs now."
Emotion_Label: ["Happy", "Neutral"],
Verdict: "KEEP",
Evidence: "I'm so happy for you!",
Reasoning: "Despite the second half being mundane filler, the initial segment contains high-intensity Joy, earning the "KEEP" status"
 
Text: "If you love someone, hold on to them. Tomorrow you might not have the chance. Sometimes you go on long trips to science centers."
Emotion_Label: ["Neutral", "Caring", "Neutral", "Anxiety", "Fear", "Neutral"],
Verdict: "KEEP",
Evidence: "Tomorrow you might not have the chance.",
Reasoning: "The mention of potential loss introduces a spike of anxiety/fear which is keep-worthy, even though the text ends with unrelated neutral content."
 
Text: "I see a chair, I see a table. Tuck the chair into the table"
Emotion_Label: ["Neutral", "Calm"],
Verdict: "DISCARD",
Evidence: "None.",
Reasoning: "Entire transcription is solely descriptive and gives an order."
 
--- CHINESE (Mandarin) ---
Text: "欢欢，还真是你啊！好久不见... 咱俩十多年没见了，你还是这么好看。走，咱俩喝酒去，走。" (Huanhuan, it's really you! Long time no see... we haven't met in ten years, you're still so pretty. Come, let's go drink, come.)
Emotion_Label: ["Surprise", "Joy", "Happy", "Warmth", "Excitement"],
Verdict: "KEEP",
Evidence: "还真是你啊！好久不见！",
Reasoning: "The text shows consistent high-energy social joy and surprise at seeing an old friend. Consistent positive emotion warrants a KEEP."

Text: "我想... 那个... 应该是明天吧。不确定。" (I think... that... should be tomorrow. Not sure.)
Emotion_Label: ["Neutral", "Hesitation", "Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "Purely informational with standard hesitation; no paralinguistic spike."

Text: "你到底想怎么样？我说过多少次了，不行就是不行！你别再来烦我了，听到没有！" (What exactly do you want? How many times have I said it, no means no! Stop bothering me, do you hear me!)
Emotion_Label: ["Anger", "Frustration", "Anger", "Annoyance", "Anger"],
Verdict: "KEEP",
Evidence: "你到底想怎么样？... 不行就是不行！",
Reasoning: "The speaker maintains a consistent high-intensity state of anger throughout. This fulfills the 'Consistent Emotion' rule."

Text: "今天天气不错，我打算去超市买点水果。苹果和香蕉都在打折。" (The weather is good today, I plan to go to the supermarket to buy some fruit. Apples and bananas are on sale.)
Emotion_Label: ["Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "The text is a simple, flat narrative of daily chores with no emotional spikes or consistent paralinguistic markers."


--- MALAY ---
Text: "Nampaknya tu, Allah ya Allah ya Allah ya Allah nawan, Allah aqbar. Wah ni, agak okey tapi terjatuh agak buntut tadi."
Emotion_Label: ["Shock", "Panic", "Pain", "Surprise", "Relief"],
Verdict: "KEEP",
Evidence: "Allah ya Allah ya Allah... terjatuh agak buntut tadi.",
Reasoning: "Multiple exclamations of 'Allah' combined with the mention of falling (pain) indicates a physical/emotional spike."

Text: "Saya rasa kita boleh bincang perkara ini esok di pejabat." (I think we can discuss this tomorrow at the office.)
Emotion_Label: ["Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "Standard professional statement with no emotional markers."

Text: "Tolonglah... saya tak tahu nak buat apa lagi... hancur semuanya... kenapa mesti jadi macam ni?" (Please... I don't know what to do anymore... everything is ruined... why must it be like this?)
Emotion_Label: ["Distress", "Despair", "Sadness", "Distress", "Despair"],
Verdict: "KEEP",
Evidence: "Hancur semuanya... kenapa mesti jadi macam ni?",
Reasoning: "Text shows consistent despair and distress. The repetitive questioning indicates high emotional weight."

Text: "Jadi, err, macam tu lah. Kita, kita kena, err, hantar laporan tu esok. Ya, esok." (So, err, that's how it is. We, we have to, err, send the report tomorrow. Yes, tomorrow.)
Emotion_Label: ["Neutral", "Hesitation", "Neutral", "Hesitation", "Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "While there is stuttering and hesitation, there is no emotional 'spike.' It is simply a non-fluent professional statement."


--- TAMIL ---
Text: "ஐயோ! என்ன இப்படி ஆகிடுச்சு? என்னால நம்பவே முடியல!" (Aiyyo! How did this happen? I can't believe it!)
Emotion_Label: ["Shock", "Distress", "Disbelief",
Verdict: "KEEP",
Evidence: "ஐயோ! என்ன இப்படி ஆகிடுச்சு?",
Reasoning: "The use of 'Aiyyo' and high-intensity disbelief indicates a clear emotional spike."

Text: "நாளைக்கு காலைல பத்து மணிக்கு நான் அங்க இருப்பேன்." (I will be there tomorrow morning at ten o'clock.)
Emotion_Label: ["Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "A simple factual statement about time and location."

Text: "யாராவது உதவி பண்ணுங்க! அங்கே ஒருத்தர் விழுந்துட்டாரு! சீக்கிரம் வாங்க!" (Somebody help! Someone fell over there! Come quickly!)
Emotion_Label: ["Panic", "Urgency", "Fear", "Urgency", "Panic"],
Verdict: "KEEP",
Evidence: "யாராவது உதவி பண்ணுங்க! சீக்கிரம் வாங்க!",
Reasoning: "High-intensity urgency and panic regarding an accident. Clear paralinguistic spike."

Text: "இந்த பெட்டியை எடுத்து அந்த மேஜை மேல வைங்க. அப்புறம் கதவை மூடிடுங்க." (Take this box and put it on that table. Then close the door.)
Emotion_Label: ["Neutral", "Calm"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "The speaker is giving a calm, descriptive set of instructions with no emotional inflection."

### TASK:
Text: '{text}'
 
### OUTPUT FORMAT:
Respond ONLY with a valid JSON object containing the keys: "Emotion_Label", "Verdict", "Evidence", and "Reasoning".

"""

def get_detailed_instruct_emotion_batch(batch_texts, prompt_template):
    formatted_prompts = []
    for text in batch_texts:
        messages = [
            {"role": "system", "content": "You are a linguistic expert specializing in emotional prosody analysis."},
            {"role": "user", "content": prompt_template.format(text=text)}
        ]
        text_ready = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text_ready)
    
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500, pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.1)
    
    responses = []
    for i in range(len(batch_texts)):
        output_ids = outputs[i][len(inputs.input_ids[i]):]
        responses.append(tokenizer.decode(output_ids, skip_special_tokens=True).strip())
    
    return responses


test_limit = None
batch_size = 4 

print(f"Found {len(neutral_files)} files to process.")


for filename in neutral_files:
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = filename.replace("_neutral.json", "_qwen_analysis.json")
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    existing_results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                temp_data = json.load(f)
                # Key by 'index' so we can check if it's already done
                existing_results = {item['index']: item for item in temp_data}
            print(f"\nResuming {filename}: {len(existing_results)} items already found.")
        except json.JSONDecodeError:
            print(f"Warning: {output_filename} was corrupted. Starting fresh for this file.")
            existing_results = {}

    with open(input_path, "r", encoding="utf-8") as f:
        full_input_data = json.load(f)

    data_to_process = [
        item for item in full_input_data 
        if item['index'] not in existing_results or 'instruct_model_results' not in existing_results[item['index']]
    ]

    if test_limit:
        data_to_process = data_to_process[:test_limit]

    if not data_to_process:
        print(f"Skipping {filename}: No new items to process based on current settings.")
        continue

    print(f"Processing {len(data_to_process)} remaining items in {filename}...")


    for i in tqdm(range(0, len(data_to_process), batch_size), desc=f"Analyzing {filename}"):
        batch_chunk = data_to_process[i:i + batch_size]
        batch_texts = [entry.get('text', "") for entry in batch_chunk]
        
        try:
            responses = get_detailed_instruct_emotion_batch(batch_texts, detailed_prompt)
            
            for j, response in enumerate(responses):
                idx = batch_chunk[j]['index']
                # Attach results to dictionary
                batch_chunk[j]['instruct_model_results'] = {
                    "full_response": response,
                    "model_used": model_name
                }
                existing_results[idx] = batch_chunk[j]

            
            final_output = sorted(existing_results.values(), key=lambda x: x['index'])
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=4, ensure_ascii=False)

            del responses
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        except torch.cuda.OutOfMemoryError:
            print(f"\n[!] CUDA Out of Memory at batch starting at index {i}. Saving and skipping to next file.")
            torch.cuda.empty_cache()
            break 
        except Exception as e:
            print(f"\n[!] Unexpected error: {e}")
            continue

print("\nAll files in the pipeline have been processed.")

