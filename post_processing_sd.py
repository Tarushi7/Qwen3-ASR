#!/usr/bin/env python3
import sys, os
import glob
import json
import numpy
from tqdm import tqdm
from datasets import load_from_disk
# ➊ Make sure Python uses your edited NeMo mixin:
sys.path.insert(0, "/home/q-wang/metadata/speaker_diarization/NeMo")

import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel

# ── Configuration ────────────────────────────────────────────────
NEMO_MODEL = "diar_sortformer_4spk-v1.nemo"
OUT_DIR    = "output"
POST_YAML  = "custom_postproc.yaml"  # or your custom_postproc.yaml if you have one
SR         = 16000
# ─────────────────────────────────────────────────────────────────


#CUDA_VISIBLE_DEVICES=3 python post_processing_sd.py

# 1) Load your patched model
print("Loading diarization model…")
model = SortformerEncLabelModel.restore_from(restore_path=NEMO_MODEL)
model.eval()

# 2) Prepare some test NumPy arrays
#    For example, two 5-second silent clips:
duration_sec = 5
silence1 = np.random.rand(duration_sec * SR).astype(np.float32)
silence2 = np.zeros(duration_sec * SR, dtype=np.float32)

#    Or replace these with any real arrays you already have:
# your_array1 = ...
# your_array2 = ...
import librosa
SR = 16000
#AUDIOs     = ["processed_batch_1413_keC7RIg3bDM_41.wav","processed_batch_10_gUFSc_MvtDg_1.wav"]
#AUDIOs     =['/data/q-wang/data/Annotation3_July2025/original_data/English_30-60sec_sg_channels1/audio/processed_batch_173_5uw5FmGCryo_142.wav', '/data/q-wang/data/Annotation3_July2025/original_data/English_30-60sec_sg_channels1/audio/processed_batch_218_7Dlvp3fzkGI_8.wav', '/data/q-wang/data/Annotation3_July2025/original_data/English_30-60sec_sg_channels1/audio/processed_batch_537_M2MyOvOmk4A_235.wav']
#AUDIO     = "processed_batch_10_gUFSc_MvtDg_1.wav"
#waveform, _ = librosa.load(AUDIO, sr = SR)

split = "train"
dur = "30_60"
dataset_names = [    
        f"Emotional-YTB-SG_en_60_{split}",
        #f"SG-ECMT-en-{split}-{dur}_v1",
        #f"SG-ECMT-zh-{split}-{dur}_v1",
        #f"SG-ECMT-ta-{split}-{dur}_v1",
        #f"SG-ECMT-ms-{split}-{dur}_v1",
]
    

for dataset_name in dataset_names:
    dataset = f"/mnt/data/datasets/yt_sea_hf_v0.2/Emotional-YTB-SG_en_30_test"
    output_path = f"/home/q-wang/server_octopus/q-wang/data/singapore_channels/test/meta/tmp/speaker"
    os.makedirs(output_path, exist_ok= True)

    arrays = []
    file_names = []
    ds = load_from_disk(dataset)
    for item in ds:
        if 'id' in item:
            file_name = item['id']
        else:
            parts = item['other_attributes']['hf_id'].split("/")
            file_name = "_".join(parts)

        file_names.append(file_name)

        wav = item['context']['audio']['array']
        wav_np = numpy.array(wav)
        #print(type(wav_np), wav_np.shape)
        arrays.append(numpy.array(wav))
        
    print("Running diarization on NumPy arrays…")
    segments_lists = model.diarize(
        audio=arrays,      # **list of np.ndarray**, no strings here
        batch_size=8,
        num_workers=4,
        postprocessing_yaml=POST_YAML,
    )
    print("Diarization complete.")
    
    for idx, segments in enumerate(tqdm(segments_lists, desc=f"Writing to jsons")):
        file_name = file_names[idx]
        json_path = os.path.join(output_path, f"{file_name}.json")
        
        id_parts = file_name.split("_")
        sub_folder = "_".join(id_parts[:3])
        id = "_".join(id_parts[3:])
        sd = []
        for seg in segments:
                
            start_time, end_time, spkid = seg.strip().split()
                            
            seg_result = {
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "speaker": spkid,
                }
            sd.append(seg_result)
                        
        # Sort by "start_time"
        sorted_sd = sorted(sd, key=lambda x: x["start_time"])

        spk_all = [x["speaker"] for x in sd]
        spk_set = set(spk_all)

        result = {
                    "file_name": file_name,
                    "num_speaker":len(spk_set),
                    "speakers": sorted_sd
                }
        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)
        
    print("\nAll done. Speaker diarizaiton json for are in:", output_path)

