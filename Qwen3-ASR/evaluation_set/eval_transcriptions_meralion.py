import os
import json
import math
import warnings
from typing import Any, Dict, List, Optional
import torch.multiprocessing as mp
import numpy as np
import torch
import glob
import librosa
from tqdm import tqdm
from datasets import Audio, Dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig


from transformers import PreTrainedModel

PreTrainedModel._supports_sdpa = False 

from transformers import PretrainedConfig
PretrainedConfig._supports_sdpa = False


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore")

REPO_ID = "MERaLiON/MERaLiON-2-10B-ASR"

DATASETS = [
    {
        "path": "/home/sailor/Tamil_eval/wav_lid_ta_dur_10_30_emotion2vec/", 
        "out_name": "ta_eval"
    },
    { 
        "path": "/home/sailor/Malay_eval/wav_lid_ms.dur_10_30.emotion2vec/",
        "out_name": "ms_eval"
    },
    {
        "path": "/home/sailor/SG_test_data/wav_lid_en_dur_10_30_emotion2vec/", 
        "out_name": "en_eval"
    },
    {
        "path": "/home/sailor/Chinese_eval/wav_lid_zh_dur_10_30_emotion2vec/", 
        "out_name": "zh_eval"
    },
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


CHUNK_LONG_AUDIO = True
CHUNK_SECONDS = 30
CHUNK_OVERLAP_SECONDS = 1


# Environment
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
    device_map="auto", 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

if not hasattr(model, "_supports_sdpa"):
    setattr(model, "_supports_sdpa", False)
    print("Manual Patch: _supports_sdpa injected successfully.")

if hasattr(model, "config") and not hasattr(model.config, "_supports_sdpa"):
    setattr(model.config, "_supports_sdpa", False)


model.eval()


# Helpers
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
    do_sample=False,
    repetition_penalty=1.1 
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
    completed_paths = set()
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                completed_paths = {item["source_audio_path"] for item in results}
            print(f"Resuming: Found {len(completed_paths)} existing records.")
        except Exception as e:
            print(f"Could not load existing file, starting fresh: {e}")
            results = []


    print("Scanning directories for .wav files...")

    audio_files = glob.glob(os.path.join(ds_path, "**", "*.wav"), recursive=True)
    audio_files.sort()
    
    if len(audio_files) == 0:
        print(f"WARNING: No wav files found in {ds_path}. Check your path structure.")
        return
    else:
        print(f"Found {len(audio_files)} .wav files to process.")

    total = TEST_LIMIT if TEST_MODE else len(audio_files)

    for global_idx in tqdm(range(total), desc=f"Dataset: {out_name}"):
        
        audio_path = audio_files[global_idx]
        
        if audio_path in completed_paths:
            continue

        audio_id = os.path.basename(audio_path).replace(".wav", "")
        
        parent_folder = os.path.basename(os.path.dirname(audio_path))
        
        try:
            audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        
            prepared_data = {
                "audio": audio, 
                "sampling_rate": TARGET_SR,
                "orig_audio_path": audio_path 
            }
            
            output = transcribe_one_audio(prepared_data["audio"])
            
            results.append({
                "idx": global_idx,
                "audio_id": audio_id,
                "emotion_subfolder": parent_folder, 
                "transcription": output["transcription"],
                "reference_text": None, 
                "duration_sec": output["duration_sec"],
                "chunked": output["chunked"],
                "num_chunks": output["num_chunks"],
                "chunk_transcriptions": output["chunk_transcriptions"],
                "source_audio_path": prepared_data["orig_audio_path"],
                "error": None,
            })
            
        except Exception as e:
            results.append({
                "idx": global_idx,
                "audio_id": audio_id,
                "emotion_subfolder": parent_folder,
                "transcription": None,
                "reference_text": None,
                "source_audio_path": audio_path,
                "error": str(e),
            })

        # Save results sorted by index 
        current_results_sorted = sorted(results, key=lambda x: x["idx"])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(safe_to_python(current_results_sorted), f, ensure_ascii=False, indent=2)

        completed_paths.add(audio_path)

    print(f"Final save completed for {out_name}: {out_path}")


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("Starting sequential processing on single GPU...")
    for item in DATASETS:
        try:
            process_dataset(item["path"], item["out_name"])
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed dataset {item['out_name']}: {e}")

if __name__ == "__main__":
    main() 

