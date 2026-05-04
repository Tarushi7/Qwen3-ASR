

import os
import json
import soundfile as sf
from datasets import load_from_disk

hf_paths = [
    "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_30_dev",
    "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_30_dev",
    "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_30_dev",
    "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_30_dev",
    "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_60_dev",
    "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_60_dev",
    "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_60_dev",
    "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ta_60_dev"
]

output_manifest = "input_manifest.json"

audio_export_dir = "./exported_audio"
os.makedirs(audio_export_dir, exist_ok=True)

with open(output_manifest, 'w') as f:

    for path in hf_paths:

        print(f"Processing dataset: {path}")

        dataset = load_from_disk(path)

        for i, entry in enumerate(dataset):

            try:
                meta = entry['other_attributes']['meta_psudo']

                diarization = meta.get('speaker_diarization', [])

                # speaker id
                if diarization and len(diarization) > 0:
                    spk_id = diarization[0].get('speaker', f"speaker_{i}")
                else:
                    spk_id = f"unknown_speaker_{i}"

                # audio
                audio_data = entry['context']['audio']
                array = audio_data['array']
                sr = audio_data['sampling_rate']

                # audio id
                audio_id = entry['other_attributes'].get('audio_id', f"clip_{i}")

                audio_filename = f"{audio_id}.wav"

                audio_path = os.path.abspath(
                    os.path.join(audio_export_dir, audio_filename)
                )

                # save wav
                sf.write(audio_path, array, sr)

                duration = len(array) / sr

                # transcription
                transcription = meta.get('transcription', "").strip()

                # split words
                if transcription == "":
                    words = ["speech", ""]
                else:
                    split_words = transcription.split()
                    words = split_words + [""]


                alignments = []

                if len(words) == 2 and words[0] == "speech":
                    alignments = [max(duration - 0.1, 0.0), duration]

                else:
                    step = duration / len(words)

                    for idx in range(len(words)):
                        align_time = min((idx + 1) * step, duration)
                        alignments.append(round(align_time, 3))

                manifest_line = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "speaker_id": spk_id,
                    "text": transcription,
                    "words": words,
                    "alignments": alignments
                }

                f.write(json.dumps(manifest_line) + "\n")

            except Exception as e:
                print(f"Skipping sample {i} from {path}")
                print("Error:", e)

print(f"\nManifest saved to: {output_manifest}")
print(f"Audio exported to: {audio_export_dir}")

