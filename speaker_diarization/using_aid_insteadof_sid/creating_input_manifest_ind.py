
#naming each speaker id with audio id to make them as "individual"
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


output_manifest = "input_manifest_aid.json"
audio_export_dir = "./exported_audio_aid"
os.makedirs(audio_export_dir, exist_ok=True)

with open(output_manifest, 'w') as f:
    for path in hf_paths:
        dataset = load_from_disk(path)
        for i, entry in enumerate(dataset):
            meta = entry['other_attributes']['meta_psudo']
            diarization = meta.get('speaker_diarization', [])

            # only grab the speaker if the list isn't empty
            if diarization and len(diarization) > 0:
                spk_id = diarization[0]['speaker']
            else:
                spk_id = f"unknown_speaker_{i}"

            audio_data = entry['context']['audio'] 
            array = audio_data['array']
            sr = audio_data['sampling_rate']
            
            audio_id = entry['other_attributes'].get('audio_id', f"clip_{i}")
            audio_filename = f"{audio_id}.wav"
            audio_path = os.path.abspath(os.path.join(audio_export_dir, audio_filename))
            sf.write(audio_path, array, sr)

            duration = len(array) / sr


            manifest_line = {
                "audio_filepath": audio_path,
                "duration": duration,
                "speaker_id": audio_id,
                "text": meta.get('transcription', ""),
                "words": [{"word": meta.get('transcription', ""), "start_time": 0.0, "end_time": duration}] 
            }
            
            f.write(json.dumps(manifest_line) + "\n")
        