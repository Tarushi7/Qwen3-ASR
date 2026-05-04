#!/usr/bin/env python3
import sys
import os
import glob
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Configuration ────────────────────────────────────────────
# Make sure this points at your patched NeMo clone if needed:
sys.path.insert(0, "/home/arjun/DiarizationExperimentation/NeMo")

from nemo.collections.asr.models import SortformerEncLabelModel

# Paths
NEMO_MODEL = "diar_sortformer_4spk-v1.nemo"
INPUT_DIR     = "input"
input_aa = "processed_batch_1413_keC7RIg3bDM_41.wav"
POST_YAML = "custom_postproc.yaml"
OUT_DIR   = "output"

# GPU devices to use
GPUS = ["cuda:0", "cuda:1"]
# ────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

def load_raw(path: str) -> np.ndarray:
    """Load a .raw PCM file (16-bit LE, mono, 16 kHz) into a float32 NumPy array."""
    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr

def init_worker(model_path: str, device: str):
    """Initialize each worker: pin its GPU and load the model once."""
    global model
    torch.cuda.set_device(device)
    model = SortformerEncLabelModel.restore_from(restore_path=model_path)
    model.to(device).eval()

import librosa
def process_file(audio_path: str, post_yaml: str, out_dir: str, device: str):
    """
    Worker function: loads one file (wav or raw), runs diarization,
    and writes out an RTTM.
    """
    import os, numpy as np, torch

    base = os.path.splitext(os.path.basename(audio_path))[0]
    ext  = os.path.splitext(audio_path)[1].lower()
    print(f"[{device}] → {base}{ext}: loading…", flush=True)

    SR = 16000
    waveform, _ = librosa.load(input_aa, sr = SR)
    # Load into the correct format
    if ext == ".raw":
        arr = load_raw(audio_path)
        audio_input = [arr]
    else:
        audio_input = [audio_path]

    # Diarize
    torch.cuda.set_device(device)
    segs = model.diarize(
        audio=[waveform],
        batch_size=1,
        num_workers=0,
        postprocessing_yaml=post_yaml,
    )[0]
    print(segs)
    exit()
    # Write RTTM
    rttm_path = os.path.join(out_dir, f"{base}.rttm")
    with open(rttm_path, "w") as fout:
        for seg in segs:
            start, dur, spk = seg.strip().split()
            fout.write(
                f"SPEAKER {base} 1 {float(start):.3f} {float(dur):.3f} "
                f"<NA> <NA> {spk} <NA> <NA>\n"
            )

    # Cleanup
    del segs
    torch.cuda.empty_cache()
    print(f"[{device}] ✔ {base}: RTTM saved to {rttm_path}", flush=True)
    return base

def main():
    # 1) Gather files
    patterns = ["*.wav", "*.raw"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(INPUT_DIR, "**", p), recursive=True))
    files = sorted(files)
    print(f"Found {len(files)} files in {INPUT_DIR}")

    if not files:
        return

    # 2) Spin up one ProcessPoolExecutor per GPU
    executors = {
        gpu: ProcessPoolExecutor(
            max_workers=1,
            initializer=init_worker,
            initargs=(NEMO_MODEL, gpu),
        )
        for gpu in GPUS
    }

    # 3) Dispatch jobs round-robin across GPUs
    futures = []
    devs = list(executors.keys())
    for idx, path in enumerate(files):
        gpu = devs[idx % len(devs)]
        futures.append(executors[gpu].submit(
            process_file, path, POST_YAML, OUT_DIR, gpu
        ))
        exit()
    # 4) Wait for all and report
    for fut in as_completed(futures):
        try:
            name = fut.result()
        except Exception as e:
            print(f"✖ Error in worker: {e}")

    # 5) Shutdown
    for exe in executors.values():
        exe.shutdown()

    print(f"\nAll done! RTTMs are in {OUT_DIR}")

if __name__ == "__main__":
    main()
