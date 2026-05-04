#!/usr/bin/env python3
import sys, os
import glob
import json
# ➊ Make sure Python uses your edited NeMo mixin:
sys.path.insert(0, "/home/q-wang/speaker_diarization/NeMo")

import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel

# ── Configuration ────────────────────────────────────────────────
NEMO_MODEL = "diar_sortformer_4spk-v1.nemo"
OUT_DIR    = "/home/tarushi/tarushi_folder/output"
POST_YAML  = "custom_postproc.yaml"  # or your custom_postproc.yaml if you have one
SR         = 16000
# ─────────────────────────────────────────────────────────────────



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
#AUDIOs     = ['/processed_batch_1413_keC7RIg3bDM_41.wav","processed_batch_10_gUFSc_MvtDg_1.wav']
#AUDIOs     =['/data/q-wang/data/Annotation3_July2025/original_data/English_30-60sec_sg_channels1/audio/processed_batch_173_5uw5FmGCryo_142.wav', '/data/q-wang/data/Annotation3_July2025/original_data/English_30-60sec_sg_channels1/audio/processed_batch_218_7Dlvp3fzkGI_8.wav', '/data/q-wang/data/Annotation3_July2025/original_data/English_30-60sec_sg_channels1/audio/processed_batch_537_M2MyOvOmk4A_235.wav']
#AUDIO     = "processed_batch_10_gUFSc_MvtDg_1.wav"
#waveform, _ = librosa.load(AUDIO, sr = SR)
AUDIOs = ['/home/tarushi/tarushi_folder/processed_batch_10_gUFSc_MvtDg_1.wav']

BASE_DIR = "/home/q-wang/server_octopus/q-wang/data/Annotation3_July2025/original_data/English_10-30sec_sg_channels1/"
IN_DIR   = os.path.join(BASE_DIR, "audio")
OUT_DIR = os.path.join(BASE_DIR, "speaker")
os.makedirs(OUT_DIR, exist_ok=True)

#AUDIOs = glob.glob(f"{IN_DIR}/*wav")
#AUDIOs = AUDIOs[:3]
files = [f.split("/")[-1][:-4] for f in AUDIOs]
print(AUDIOs)
arrays = []
for AUDIO in AUDIOs:
    waveform, _ = librosa.load(AUDIO, sr = SR)
    arrays.append(waveform)
    print(type(waveform),waveform.shape)

    print(type(arrays))
    exit()
print(f"Prepared {len(arrays)} NumPy arrays for diarization.")

# 3) Run diarization on the list of arrays
#exit(arrays)
print("Running diarization on NumPy arrays…")
segments_lists = model.diarize(
    audio=arrays,      # **list of np.ndarray**, no strings here
    batch_size=1,
    num_workers=0,
    postprocessing_yaml=POST_YAML,
)
print("Diarization complete.")
print(segments_lists)

for idx, segments in enumerate(segments_lists):
    file_name = files[idx]
    json_path = os.path.join(OUT_DIR, f"{file_name}.json")
    
    id_parts = file_name.split("_")
    sub_folder = "_".join(id_parts[:3])
    id = "_".join(id_parts[3:])
    sd = []
    for seg in segments:
            
        start_time, end_time, spkid = seg.strip().split()
                        
        seg_result = {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "spk_id": spkid,
            }
        sd.append(seg_result)
                    
    # Sort by "start_time"
    sorted_sd = sorted(sd, key=lambda x: x["start_time"])

    spk_all = [x["spk_id"] for x in sd]
    spk_set = set(spk_all)

    result = {
                "file_name": file_name,
                "sub_folder": sub_folder,
                "id": id,
                "num_speaker":len(spk_set),
                "speaker_diarization": sorted_sd
            }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)
print("\nAll done. Speaker diarizaiton json for are in:", OUT_DIR)
exit()
# 4) Write out one RTTM per array
for idx, segments in enumerate(segments_lists, start=1):
    base = f"np_array_{idx}"
    rttm_path = os.path.join(OUT_DIR, f"{base}.rttm")
    with open(rttm_path, "w") as fout:
        for seg in segments:
            start, dur, spk = seg.strip().split()
            fout.write(
                f"SPEAKER {base} 1 {float(start):.3f} {float(dur):.3f} "
                f"<NA> <NA> {spk} <NA> <NA>\n"
            )
    print(f"✅ Wrote RTTM → {rttm_path}")

print("\nAll done. RTTMs for NumPy arrays are in:", OUT_DIR)


def convert_sd_to_fixlen(segments_lists):              
    sd_spk = []
    for seg in segments_lists[0]:
        start_time, end_time, spkid = seg.strip().split()
        sd_spk.append([float(start_time), float(end_time), spkid])

    max_time = max(end for _, end, _ in sd_spk)
    seg_len = 2
    windows = [(start, start+seg_len) for start in range(0, int(max_time)+seg_len, seg_len)]

    result = []
    for w_start, w_end in windows:
        speakers = []
        for s_start, s_end, spk in sd_spk:
            # overlap check: [s_start, s_end) intersects [w_start, w_end)
            if s_start < w_end and s_end > w_start:
                speakers.append(spk)
        if speakers:
            result.append([w_start, w_end] + sorted(set(speakers)))
        else:
            result.append([w_start, w_end, "na"])
    print(result)
    return result