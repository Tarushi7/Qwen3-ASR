import os
import json
import math
import warnings
from typing import Any, Dict, List, Optional
import librosa
import torch.multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
REPO_ID = "MERaLiON/MERaLiON-2-10B-ASR"

DATASETS = [
    # {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_30_dev", "out_name": "en_30"},
    # {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_30_dev", "out_name": "zh_30"},
    # {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_30_dev", "out_name": "ms_30"},
    # {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_30_dev", "out_name": "ta_30"},
    # {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_60_dev", "out_name": "en_60"},
    # {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_60_dev", "out_name": "zh_60"},
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_60_dev", "out_name": "ms_60"},
    # {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_60_dev", "out_name": "ta_60"},
]

OUTPUT_DIR = "./meralion_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_MODE = False 
TEST_LIMIT = 4

BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
TARGET_SR = 16000

PROMPT_TEMPLATE = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"
TRANSCRIBE_PROMPT = PROMPT_TEMPLATE.format(query="Please transcribe this speech.")

CHUNK_LONG_AUDIO = True
CHUNK_SECONDS = 30
CHUNK_OVERLAP_SECONDS = 1

# =========================
# Helpers
# =========================
def safe_to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict): return {k: safe_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_to_python(x) for x in obj]
    return obj

def extract_audio_info(sample: Dict[str, Any]) -> Dict[str, Any]:
    context = sample.get("context", {})
    audio_info = context.get("audio", None)
    if audio_info is None or not isinstance(audio_info, dict):
        raise ValueError("audio_info is missing")
    audio_array = np.asarray(audio_info.get("array"), dtype=np.float32)
    sampling_rate = audio_info.get("sampling_rate")
    return {"array": audio_array, "sampling_rate": int(sampling_rate), "path": audio_info.get("path")}

def mono_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1: return audio
    return audio.mean(axis=0 if audio.shape[0] <= 8 else 1).astype(np.float32)

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(audio.astype(np.float32))
    max_abs = np.max(np.abs(audio)) if audio.size > 0 else 0.0
    if max_abs > 1.0: audio /= max_abs
    return audio

def resample_if_needed(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR: return audio.astype(np.float32)
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

def prepare_audio_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    info = extract_audio_info(sample)
    audio = resample_if_needed(normalize_audio(mono_audio(info["array"])), info["sampling_rate"])
    return {"audio": audio, "sampling_rate": TARGET_SR, "orig_audio_path": info["path"]}

def split_audio(audio: np.ndarray, sr: int) -> List[np.ndarray]:
    chunk_size, overlap = int(CHUNK_SECONDS * sr), int(CHUNK_OVERLAP_SECONDS * sr)
    if audio.size <= chunk_size: return [audio]
    chunks, step, start = [], chunk_size - overlap, 0
    while start < audio.size:
        end = start + chunk_size
        chunk = audio[start:end]
        if chunk.size == 0: break
        chunks.append(chunk)
        if end >= audio.size: break
        start += step
    return chunks

@torch.inference_mode()
def transcribe_batch(audio_list: List[np.ndarray]) -> List[str]:
    if not audio_list: return []
    prompts = [[{"role": "user", "content": TRANSCRIBE_PROMPT}]] * len(audio_list)
    text_input = processor.tokenizer.apply_chat_template(conversation=prompts, tokenize=False, add_generation_prompt=True)
    if isinstance(text_input, str): text_input = [text_input] * len(audio_list)

    inputs = processor(text=text_input, audios=audio_list, sampling_rate=TARGET_SR)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device).to(torch.bfloat16) if v.dtype == torch.float32 else v.to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return [t.strip() for t in processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]

def transcribe_one_audio(audio: np.ndarray) -> Dict[str, Any]:
    dur = len(audio) / TARGET_SR
    if CHUNK_LONG_AUDIO and dur > CHUNK_SECONDS:
        chunks = split_audio(audio, TARGET_SR)
        texts = [transcribe_batch([ch])[0] for ch in chunks]
        return {"transcription": " ".join([t for t in texts if t]).strip(), "duration_sec": dur, "chunked": True, "num_chunks": len(chunks), "chunk_transcriptions": texts}
    t = transcribe_batch([audio])[0]
    return {"transcription": t, "duration_sec": dur, "chunked": False, "num_chunks": 1, "chunk_transcriptions": [t]}

def process_dataset(ds_path: str, out_name: str) -> None:
    print(f"\nGPU {torch.cuda.current_device()} Processing: {ds_path}")
    out_path = os.path.join(OUTPUT_DIR, f"{out_name}_transcriptions.json")
    results, completed_indices = [], set()
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                completed_indices = {item["idx"] for item in results}
        except: pass

    ds = load_from_disk(ds_path)
    try: ds = ds.cast_column("context.audio", Audio(sampling_rate=TARGET_SR))
    except: pass
    
    total = TEST_LIMIT if TEST_MODE else len(ds)
    for start in tqdm(range(0, total, BATCH_SIZE), desc=f"DS: {out_name}"):
        end_idx = min(start + BATCH_SIZE, total)
        batch_range = range(start, end_idx)
        if all(idx in completed_indices for idx in batch_range): continue

        batch_samples = [ds[i] for i in batch_range]
        for i, sample in enumerate(batch_samples):
            g_idx = start + i
            try:
                prep = prepare_audio_from_sample(sample)
                out = transcribe_one_audio(prep["audio"])
                results.append({"idx": g_idx, "audio_id": sample.get("other_attributes",{}).get("audio_id",f"s_{g_idx}"), "transcription": out["transcription"], "duration_sec": out["duration_sec"], "error": None})
            except Exception as e:
                results.append({"idx": g_idx, "error": str(e)})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(safe_to_python(sorted(results, key=lambda x: x["idx"])), f, ensure_ascii=False, indent=2)

# =========================
# Multiprocessing Worker
# =========================
def worker(gpu_id: int, tasks: List[Dict]):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    global model, processor
    processor = AutoProcessor.from_pretrained(REPO_ID, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(REPO_ID, quantization_config=quant_config, device_map={"": 0}, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    if not hasattr(model, "_supports_sdpa"): setattr(model, "_supports_sdpa", False)

    for task in tasks:
        process_dataset(task["path"], task["out_name"])
        torch.cuda.empty_cache()

def main():
    mp.set_start_method("spawn", force=True)
    
    # Split datasets: Even indices to GPU 0, Odd to GPU 1
    gpu0_tasks = DATASETS[::2]
    gpu1_tasks = DATASETS[1::2]

    p0 = mp.Process(target=worker, args=(0, gpu0_tasks))
    p1 = mp.Process(target=worker, args=(1, gpu1_tasks))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

if __name__ == "__main__":
    main()

