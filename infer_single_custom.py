
import os
from nemo.collections.asr.models import SortformerEncLabelModel
import librosa

# Paths – adjust as needed
NEMO_MODEL = "diar_sortformer_4spk-v1.nemo"
AUDIO     = "processed_batch_1413_keC7RIg3bDM_41.wav"
POST_YAML = "custom_postproc.yaml"
OUT_DIR   = "output"

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load pretrained Sortformer diarizer
model = SortformerEncLabelModel.restore_from(restore_path=NEMO_MODEL)
model.eval()

SR = 16000
waveform, _ = librosa.load(AUDIO, sr = SR)

# 2) Run diarization (inference + dynamic clustering + post-processing)
#    No need to specify number of speakers; clustering auto-estimates it.
segments_list = model.diarize(
    [waveform],                      # positional list of files
    batch_size=1,
    num_workers=1,
    postprocessing_yaml=POST_YAML, # pass YAML path directly
)
segments = segments_list[0]

# 3) Write out RTTM
basename = os.path.splitext(os.path.basename(AUDIO))[0]
rttm_path = os.path.join(OUT_DIR, f"{basename}.rttm")
with open(rttm_path, 'w') as fout:
    for seg in segments:
        # seg is a string: '<start> <duration> <speaker>'
        start_str, dur_str, speaker = seg.strip().split()
        start = float(start_str)
        dur   = float(dur_str)
        fout.write(
            f"SPEAKER {basename} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        )

print(f"✅ RTTM written → {rttm_path}")


