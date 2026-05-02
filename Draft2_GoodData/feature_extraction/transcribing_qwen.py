import os
import json
import torch
import numpy as np
import torch.multiprocessing as mp
from datasets import load_from_disk, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig
import transformers

def process_meralion_task(gpu_id, task_list, output_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    model_id = "MERaLiON/MERaLiON-2-10B-ASR"
    
    transformers.PreTrainedModel._supports_sdpa = False

    print(f"GPU {gpu_id}: Loading model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    for task in task_list:
        path = task["path"]
        out_name = task["out_name"]
        print(f"\n--- GPU {gpu_id} START: {out_name} ---")

        ds = load_from_disk(path)
        
        # We check which keys exist to avoid the "missing dictionary" error
        all_cols = ds.column_names
        audio_key = None
        
        # Priority 1: Check if it's already flattened
        if "context.audio" in all_cols:
            audio_key = "context.audio"
        # Priority 2: Check for raw context to flatten it
        elif "context" in all_cols:
            ds = ds.flatten()
            # After flattening, the key usually becomes 'context.audio'
            audio_key = "context.audio"
        
        if audio_key:
            try:
                ds = ds.cast_column(audio_key, Audio(sampling_rate=16000))
            except:
                pass 

        results = []

        for i in range(len(ds)):
            try:
                item = ds[i]
                
                # --- FLEXIBLE AUDIO RETRIEVAL ---
                audio_info = item.get(audio_key) or item.get("audio")
                
                if audio_info is None or not isinstance(audio_info, dict):
                    # Fallback: if flattened context.audio doesn't work, try nested
                    if "context" in item and isinstance(item["context"], dict):
                        audio_info = item["context"].get("audio")

                if audio_info is None or not isinstance(audio_info, dict):
                    raise ValueError(f"Could not find audio dict. Available keys: {list(item.keys())}")

                audio_array = audio_info.get("array")
                sampling_rate = audio_info.get("sampling_rate", 16000)

                if audio_array is None:
                    raise ValueError("audio_array is None")

                audio_np = np.array(audio_array, dtype=np.float32)
                
                # Standard cleaning
                audio_np = np.nan_to_num(audio_np)
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()

                # Inference
                inputs = processor(
                    text="Please transcribe the speech in the audio.",
                    audio=audio_np,
                    sampling_rate=sampling_rate,
                    return_tensors="pt"
                )

                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=256)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                if i % 10 == 0:
                    print(f"GPU {gpu_id} | {i} | {text[:50]}")

                results.append({"index": i, "text": text, "source": "asr"})

            except Exception as e:
                results.append({"index": i, "text": "", "error": str(e)})

        # Save output
        output_path = os.path.join(output_dir, f"{out_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        torch.cuda.empty_cache()

    print(f"GPU {gpu_id} DONE")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    output_dir = "/home/tarushi/tarushi_folder/Qwen3-ASR/Draft2_GoodData/feature_extraction"
    os.makedirs(output_dir, exist_ok=True)

    all_tasks = [
        {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_30_dev", "out_name": "en_30"},
        {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_30_dev", "out_name": "zh_30"},
        {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_30_dev", "out_name": "ms_30"},
        {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_30_dev", "out_name": "ta_30"},
        {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_60_dev", "out_name": "en_60"},
        {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_60_dev", "out_name": "zh_60"},
        {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_60_dev", "out_name": "ms_60"},
        {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_60_dev", "out_name": "ta_60"},
    ]

    gpu0_tasks = all_tasks[::2]
    gpu1_tasks = all_tasks[1::2]

    p0 = mp.Process(target=process_meralion_task, args=(0, gpu0_tasks, output_dir))
    p1 = mp.Process(target=process_meralion_task, args=(1, gpu1_tasks, output_dir))

    p0.start()
    p1.start()
    p0.join()
    p1.join()
    print("Workflow complete.")