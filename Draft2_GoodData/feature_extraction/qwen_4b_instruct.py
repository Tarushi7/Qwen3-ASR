# import json
# import os
# import torch
# import re
# from transformers import AutoModelForCausalLM, AutoTokenizer


# model_name = "Qwen/Qwen3-4B-Instruct-2507"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# # Prompt
# detailed_prompt = """

# PROMPT 4:
  
# ### INSTRUCTION:
# You are a high-precision linguistic filter for a paralinguistic dataset.
# CRITICAL NOTE: Transcriptions may contain sudden, disjointed shifts in emotion or topic. A file might contain 90percent filler and 10percent intense emotion. If AT LEAST ONE phrase shows significant emotional weight (Anger, Panic, Joy, Distress, Fear, Surprise, Confusion, Disgust, Empathy, etc.), the file must be marked as KEEP. We are strictly filtering for emotional "spikes" regardless of the surrounding neutral context.
 
# ### RULES:
# 1. Look at each phrase (roughly every 4-6 words).
# 2. Assign an emotion label to EACH segment as a list. Use 'Neutral' for flat text and specific labels for emotions.
# 3. Identify the specific phrase (Evidence) that justifies the highest emotional intensity.
# 4. Final Verdict: KEEP if you sense any paralinguistic spike in one or more of the phrases. Otherwise, DISCARD.
# 5. Only decide KEEP or DISCARD after looking at emotion labels
 
# ### EXAMPLES:
# Text: "I'll see you at five. Oh my god! Is that a snake? Get back!"
# Emotion_Label: ["Neutral", "Neutral", "Surprise", "Fearful", "Fearful", "Panic"],
# Verdict: "KEEP",
# Evidence: "Oh my god! Is that a snake? Get back!",
# Reasoning: "Text starts as a neutral appointment but shifts abruptly to life-threatening panic. The presence of 'Panic' labels requires a KEEP verdict."
 
# Text: "The data shows a slight increase in margin. We should review this next Tuesday."
# Emotion_Label: ["Neutral", "Neutral", "Neutral", "Neutral", "Neutral"],
# Verdict: "DISCARD",
# Evidence: "None",
# Reasoning: "Entire transcription is professional, informational, and lacks any emotional phrases."
 
# Text: "I'm so happy for you! Anyway, I need to go buy some milk and eggs now."
# Emotion_Label: ["Happy", "Happy", "Happy", "Neutral", "Neutral", "Neutral", "Neutral"],
# Verdict: "KEEP",
# Evidence: "I'm so happy for you!",
# Reasoning: "Despite the second half being mundane filler, the initial segment contains high-intensity Joy, earning the "KEEP" status"
 
# Text: "If you love someone, hold on to them. Tomorrow you might not have the chance. Sometimes you go on long trips to science centers."
# Emotion_Label: ["Neutral", "Caring", "Neutral", "Anxiety", "Fear", "Neutral", "Neutral", "Neutral"],
# Verdict: "KEEP",
# Evidence: "Tomorrow you might not have the chance.",
# Reasoning: "The mention of potential loss introduces a spike of anxiety/fear which is keep-worthy, even though the text ends with unrelated neutral content."
 
# Text: "I see a chair, I see a table. Tuck the chair into the table"
# Emotion_Label: ["Neutral", "Neutral", "Calmness"],
# Verdict: "DISCARD",
# Evidence: "None.",
# Reasoning: "Entire transcription is solely descriptive and gives an order."
 
# ### TASK:
# Text: '{text}'
 
# ### OUTPUT FORMAT:
# Respond ONLY with a valid JSON object containing the keys: "Emotion_Label", "Verdict", "Evidence", and "Reasoning".

# """

# def get_detailed_instruct_emotion(text, prompt_template):
#     messages = [
#         {"role": "system", "content": "You are a linguistic expert specializing in emotional prosody analysis."},
#         {"role": "user", "content": prompt_template.format(text=text)}
#     ]
#     input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    
#     # max_new_tokens=512 is the sweet spot for reasoning + labels
#     outputs = model.generate(**inputs, max_new_tokens=1024)
#     output_ids = outputs[0][len(inputs.input_ids[0]):]
#     return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

# # 3. FILE PROCESSING
# input_file = "ml_dataset_neutral.json"
# output_file = "ml_dataset_instruct_analysis.json"
# test_limit = 50

# if os.path.exists(input_file):
#     with open(input_file, "r", encoding="utf-8") as f:
#         full_data = json.load(f)

#     # Slice the data AND the texts to process immediately
#     data = full_data[:test_limit]
#     texts_to_process = [entry.get('text', "") for entry in data]

#     print(f"Starting analysis on {len(data)} files...")

#     for i, text in enumerate(texts_to_process):
#         # Safety check: skip if ASR failed to produce text
#         if not text.strip():
#             data[i]['instruct_model_results'] = {"full_response": "N/A - Empty Text"}
#             continue

#         # Run model
#         response = get_detailed_instruct_emotion(text, detailed_prompt)
        
#         # Save results back to our 'data' slice
#         data[i]['instruct_model_results'] = {
#             "full_response": response,
#             "model_used": model_name
#         }
        
#         # Proactive Progress Tracker
#         print(f"[{i+1}/{test_limit}] Processed Index: {data[i]['index']}")

#     # 4. FINAL SAVE
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4)
    
#     print(f"Results saved to {output_file}")

# else:
#     print(f"Error: {input_file} not found. Check your file path!")





#with batching:
import json
import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- BATCHING SETUP ---
tokenizer.padding_side = "left" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Prompt
detailed_prompt = """

### INSTRUCTION:
You are a high-precision linguistic filter for a paralinguistic dataset.
CRITICAL NOTE: Transcriptions may contain sudden, disjointed shifts in emotion or topic. A file might contain 90 percent filler and 10 percent intense emotion. If AT LEAST ONE phrase shows significant emotional weight (Anger, Panic, Joy, Distress, Fear, Surprise, Confusion, Disgust, Empathy, etc.), the file must be marked as KEEP. We are strictly filtering for emotional phrases regardless of the surrounding neutral context.

 
### RULES:
1. Look at each phrase (roughly every 4-6 words).
2. Assign an emotion label to EACH segment as a list. Use 'Neutral' for flat text and specific labels for emotions.
3. Identify the specific phrase (Evidence) that justifies the highest emotional intensity.
4. Final Verdict: KEEP if you sense any paralinguistic spike in one or more of the phrases. Otherwise, DISCARD.
5. If a specific emotion label is consistent throughout the entire file (e.g., the speaker is crying or angry from start to finish), you must also mark it as KEEP. Do not discard just because there is no variation.
6. Only decide KEEP or DISCARD after looking at emotion labels
 

### MULTILINGUAL EXAMPLES:
--- English ---
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
Emotion_Label: ["Neutral", "Neutral", "Neutral", "Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "The text is a simple, flat narrative of daily chores with no emotional spikes or consistent paralinguistic markers."


--- MALAY ---
Text: "Nampaknya tu, Allah ya Allah ya Allah ya Allah nawan, Allah aqbar. Wah ni, agak okey tapi terjatuh agak buntut tadi."
Emotion_Label: ["Shock", "Panic", "Pain", "Pain", "Surprise", "Relief"],
Verdict: "KEEP",
Evidence: "Allah ya Allah ya Allah... terjatuh agak buntut tadi.",
Reasoning: "Multiple exclamations of 'Allah' combined with the mention of falling (pain) indicates a physical/emotional spike."

Text: "Saya rasa kita boleh bincang perkara ini esok di pejabat." (I think we can discuss this tomorrow at the office.)
Emotion_Label: ["Neutral", "Neutral", "Neutral"],
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
Emotion_Label: ["Shock", "Distress", "Disbelief", "Disbelief"],
Verdict: "KEEP",
Evidence: "ஐயோ! என்ன இப்படி ஆகிடுச்சு?",
Reasoning: "The use of 'Aiyyo' and high-intensity disbelief indicates a clear emotional spike."

Text: "நாளைக்கு காலைல பத்து மணிக்கு நான் அங்க இருப்பேன்." (I will be there tomorrow morning at ten o'clock.)
Emotion_Label: ["Neutral", "Neutral", "Neutral"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "A simple factual statement about time and location."

Text: "யாராவது உதவி பண்ணுங்க! அங்கே ஒருத்தர் விழுந்துட்டாரு! சீக்கிரம் வாங்க!" (Somebody help! Someone fell over there! Come quickly!)
Emotion_Label: ["Panic", "Urgency", "Fear", "Urgency", "Panic"],
Verdict: "KEEP",
Evidence: "யாராவது உதவி பண்ணுங்க! சீக்கிரம் வாங்க!",
Reasoning: "High-intensity urgency and panic regarding an accident. Clear paralinguistic spike."

Text: "இந்த பெட்டியை எடுத்து அந்த மேஜை மேல வைங்க. அப்புறம் கதவை மூடிடுங்க." (Take this box and put it on that table. Then close the door.)
Emotion_Label: ["Neutral", "Neutral", "Neutral", "Calm"],
Verdict: "DISCARD",
Evidence: "None",
Reasoning: "The speaker is giving a calm, descriptive set of instructions with no emotional inflection."

### TASK:
Text: '{text}'
 
### OUTPUT FORMAT:
Respond ONLY with a valid JSON object containing the keys: "Emotion_Label", "Verdict", "Evidence", and "Reasoning".

"""


def get_detailed_instruct_emotion_batch(batch_texts, prompt_template):
    # Prepare all messages in the batch
    formatted_prompts = []
    for text in batch_texts:
        messages = [
            {"role": "system", "content": "You are a linguistic expert specializing in emotional prosody analysis."},
            {"role": "user", "content": prompt_template.format(text=text)}
        ]
        text_ready = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text_ready)
    
    # Tokenize the entire batch at once
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(model.device)
    
    # Generate for the whole batch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id)
    
    # Extract only the generated parts
    responses = []
    for i in range(len(batch_texts)):
        output_ids = outputs[i][len(inputs.input_ids[i]):]
        responses.append(tokenizer.decode(output_ids, skip_special_tokens=True).strip())
    
    return responses

# 3. FILE PROCESSING
input_file = "ml_dataset_neutral.json"
output_file = "ml_dataset_instruct_analysis.json"
test_limit = 100
batch_size = 10

if os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    data = full_data[:test_limit]
    print(f"Starting batch analysis (Size: {batch_size}) on {len(data)} files...")

    # Loop through data in chunks of batch_size
    for i in range(0, len(data), batch_size):
        batch_chunk = data[i:i + batch_size]
        batch_texts = [entry.get('text', "") for entry in batch_chunk]
        
        # Run batch model inference
        responses = get_detailed_instruct_emotion_batch(batch_texts, detailed_prompt)
        
        # Save results back to the original data objects
        for j, response in enumerate(responses):
            batch_chunk[j]['instruct_model_results'] = {
                "full_response": response,
                "model_used": model_name
            }
        
        print(f"Processed up to index: {i + len(batch_chunk)}")

    # 4. FINAL SAVE
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to {output_file}")

else:
    print(f"Error: {input_file} not found. Check your file path!")