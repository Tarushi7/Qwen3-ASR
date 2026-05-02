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
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_30_dev", "out_name": "en_30"},
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_30_dev", "out_name": "zh_30"},
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_30_dev", "out_name": "ms_30"},
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_30_dev", "out_name": "ta_30"},
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_60_dev", "out_name": "en_60"},
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_60_dev", "out_name": "zh_60"},
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_60_dev", "out_name": "ms_60"},
    {"path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_60_dev", "out_name": "ta_60"},
]

OUTPUT_DIR = "./meralion_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_MODE = False 
TEST_LIMIT = 2

BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
TARGET_SR = 16000

PROMPT_TEMPLATE = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"
TRANSCRIBE_PROMPT = PROMPT_TEMPLATE.format(query="Please transcribe this speech.")

# If you want to chunk >30s audio for safer ASR, keep this True.
CHUNK_LONG_AUDIO = True
CHUNK_SECONDS = 30
CHUNK_OVERLAP_SECONDS = 1

# =========================
# Environment
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = device == "cuda"

print(f"Using device: {device}")


try:
    import transformers
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False
except Exception as e:
    print(f"Warning while patching _supports_sdpa: {e}")


# =========================
# Load processor/model 
# =========================
from transformers import BitsAndBytesConfig

# Clear cache before starting
torch.cuda.empty_cache()

processor = AutoProcessor.from_pretrained(REPO_ID, trust_remote_code=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"Loading model in 4-bit on {device}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    REPO_ID,
    quantization_config=quant_config,
    device_map="auto", # Let transformers decide the best placement
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.eval()

# Patch _supports_sdpa
if not hasattr(model, "_supports_sdpa"):
    setattr(model, "_supports_sdpa", False)



# =========================
# Helpers
# =========================
def safe_to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: safe_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_to_python(x) for x in obj]
    return obj

def extract_audio_info(sample: Dict[str, Any]) -> Dict[str, Any]:

    context = sample.get("context", {})
    audio_info = context.get("audio", None)

    if audio_info is None or not isinstance(audio_info, dict):
        raise ValueError("audio_info is missing or not a dictionary")

    audio_array = audio_info.get("array", None)
    sampling_rate = audio_info.get("sampling_rate", None)
    audio_path = audio_info.get("path", None)

    if audio_array is None:
        raise ValueError("audio array is missing")

    audio_array = np.asarray(audio_array, dtype=np.float32)

    if audio_array.ndim == 0:
        raise ValueError("audio array is scalar, invalid")
    if audio_array.size == 0:
        raise ValueError("audio array is empty")

    if sampling_rate is None:
        raise ValueError("sampling_rate is missing")

    return {
        "array": audio_array,
        "sampling_rate": int(sampling_rate),
        "path": audio_path,
    }

def mono_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:

        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return audio.mean(axis=0).astype(np.float32)
        return audio.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim: {audio.ndim}")

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = np.max(np.abs(audio)) if audio.size > 0 else 0.0
    if max_abs > 1.0:
        audio = audio / max_abs
    return audio

def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)

    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)

def prepare_audio_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    info = extract_audio_info(sample)
    audio = info["array"]
    sr = info["sampling_rate"]

    audio = mono_audio(audio)
    audio = normalize_audio(audio)
    audio = resample_if_needed(audio, sr, TARGET_SR)

    if audio is None:
        raise ValueError("Processed audio is None")
    if not isinstance(audio, np.ndarray):
        raise ValueError("Processed audio is not a numpy array")
    if audio.ndim != 1:
        raise ValueError(f"Processed audio must be 1D, got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("Processed audio is empty")

    return {
        "audio": audio,
        "sampling_rate": TARGET_SR,
        "orig_audio_path": info["path"],
    }

def split_audio(audio: np.ndarray, sr: int, chunk_seconds: int = 30, overlap_seconds: int = 1) -> List[np.ndarray]:
    chunk_size = int(chunk_seconds * sr)
    overlap = int(overlap_seconds * sr)

    if audio.size <= chunk_size:
        return [audio]

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < audio.size:
        end = start + chunk_size
        chunk = audio[start:end]
        if chunk.size == 0:
            break
        chunks.append(chunk)
        if end >= audio.size:
            break
        start += step
    return chunks

def build_chat_prompts(batch_size: int) -> List[List[Dict[str, str]]]:
    return [[{"role": "user", "content": TRANSCRIBE_PROMPT}] for _ in range(batch_size)]

@torch.inference_mode()
def transcribe_batch(audio_list: List[np.ndarray]) -> List[str]:
    if len(audio_list) == 0:
        return []

    conversation = build_chat_prompts(len(audio_list))
    
    text_input = processor.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    if isinstance(text_input, str):
        text_input = [text_input] * len(audio_list)


    inputs = processor(
        text=text_input,
        audios=audio_list,
        sampling_rate=TARGET_SR
    )


    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(model.device) 
            if inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(torch.bfloat16)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False 
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return [t.strip() for t in texts]


def transcribe_one_audio(audio: np.ndarray) -> Dict[str, Any]:
    duration_sec = len(audio) / TARGET_SR

    if CHUNK_LONG_AUDIO and duration_sec > CHUNK_SECONDS:
        chunks = split_audio(audio, TARGET_SR, CHUNK_SECONDS, CHUNK_OVERLAP_SECONDS)
        chunk_texts = []
        for ch in chunks:
            text = transcribe_batch([ch])[0]
            chunk_texts.append(text)

        merged_text = " ".join([t for t in chunk_texts if t]).strip()
        return {
            "transcription": merged_text,
            "chunked": True,
            "num_chunks": len(chunks),
            "duration_sec": duration_sec,
            "chunk_transcriptions": chunk_texts,
        }

    text = transcribe_batch([audio])[0]
    return {
        "transcription": text,
        "chunked": False,
        "num_chunks": 1,
        "duration_sec": duration_sec,
        "chunk_transcriptions": [text],
    }

def get_reference_text(sample: Dict[str, Any]) -> Optional[str]:
    try:
        return sample.get("other_attributes", {}).get("meta_psudo", {}).get("transcription", None)
    except Exception:
        return None

def get_audio_id(sample: Dict[str, Any], idx: int) -> str:
    try:
        return sample.get("other_attributes", {}).get("audio_id", f"sample_{idx}")
    except Exception:
        return f"sample_{idx}"

def process_dataset(ds_path: str, out_name: str) -> None:
    print(f"\nProcessing dataset: {ds_path}")
    out_path = os.path.join(OUTPUT_DIR, f"{out_name}_transcriptions.json")
    
    results = []
    completed_indices = set()
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                completed_indices = {item["idx"] for item in results}
            print(f"Resuming: Found {len(completed_indices)} existing records.")
        except Exception as e:
            print(f"Could not load existing file, starting fresh: {e}")
            results = []

    ds = load_from_disk(ds_path)
    try:
        ds = ds.cast_column("context.audio", Audio(sampling_rate=TARGET_SR))
    except Exception as e:
        print(f"Skipping cast_column: {e}")

    total = TEST_LIMIT if TEST_MODE else len(ds)

    # Wrap the range in tqdm for a progress bar
    for start in tqdm(range(0, total, BATCH_SIZE), desc=f"Dataset: {out_name}"):
        end_idx = min(start + BATCH_SIZE, total)
        
        batch_range = range(start, end_idx)
        if all(idx in completed_indices for idx in batch_range):
            continue

        batch_samples = [ds[i] for i in batch_range]
        prepared_items = []
        error_items = []

        for local_idx, sample in enumerate(batch_samples):
            global_idx = start + local_idx
            try:
                prepared = prepare_audio_from_sample(sample)
                prepared_items.append((global_idx, sample, prepared))
            except Exception as e:
                error_items.append({
                    "idx": global_idx,
                    "audio_id": get_audio_id(sample, global_idx),
                    "error": str(e),
                    "transcription": None,
                    "reference_text": get_reference_text(sample),
                })

        results.extend(error_items)
        
        for global_idx, sample, prepared in prepared_items:
            try:
                output = transcribe_one_audio(prepared["audio"])
                results.append({
                    "idx": global_idx,
                    "audio_id": get_audio_id(sample, global_idx),
                    "transcription": output["transcription"],
                    "reference_text": get_reference_text(sample),
                    "duration_sec": output["duration_sec"],
                    "chunked": output["chunked"],
                    "num_chunks": output["num_chunks"],
                    "chunk_transcriptions": output["chunk_transcriptions"],
                    "source_audio_path": prepared["orig_audio_path"],
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "idx": global_idx,
                    "audio_id": get_audio_id(sample, global_idx),
                    "transcription": None,
                    "reference_text": get_reference_text(sample),
                    "source_audio_path": prepared["orig_audio_path"],
                    "error": str(e),
                })

        current_results_sorted = sorted(results, key=lambda x: x["idx"])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(safe_to_python(current_results_sorted), f, ensure_ascii=False, indent=2)

        for idx in batch_range:
            completed_indices.add(idx)

    print(f"Final save completed for {out_name}: {out_path}")

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("Starting sequential processing on single GPU...")
    for item in DATASETS:
        try:
            process_dataset(item["path"], item["out_name"])
            # Clear memory between different datasets
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed dataset {item['out_name']}: {e}")

if __name__ == "__main__":
    main() 



