import json
import os
import soundfile as sf
from datasets import load_from_disk


DATA_MAPPING = [
    {
        "hf_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_en_30_dev",
        "json_path": "/home/tarushi/tarushi_folder/speaker_diarization_qwen_transcripts/30s_transcriptions_qwen/en_30_transcriptions_qwen.json",
        "output_dir": "/home/tarushi/tarushi_folder/speaker_diarization/extracted_wavs/en"
    },
    {
        "hf_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_zh_30_dev",
        "json_path": "/home/tarushi/tarushi_folder/speaker_diarization_qwen_transcripts/30s_transcriptions_qwen/cn_30_transcriptions_qwen.json",
        "output_dir": "/home/tarushi/tarushi_folder/speaker_diarization/extracted_wavs/zh"
    },
    {
        "hf_path": "/home/q-wang/data/hf_egs/Emotional-YTB-MY_ms_30_dev",
        "json_path": "/home/tarushi/tarushi_folder/speaker_diarization_qwen_transcripts/30s_transcriptions_qwen/ml_30_transcriptions_qwen.json",
        "output_dir": "/home/tarushi/tarushi_folder/speaker_diarization/extracted_wavs/ms"
    }
]

FINAL_MANIFEST = "qwen_input_manifest.jsonl"

def process():
    manifest_entries = []

    for config in DATA_MAPPING:
        if not os.path.exists(config["hf_path"]):
            print(f"Skipping: HF Path not found {config['hf_path']}")
            continue
            
        ds = load_from_disk(config["hf_path"])
        os.makedirs(config["output_dir"], exist_ok=True)
        
       
        with open(config["json_path"], 'r') as f:
            transcripts = json.load(f)

        print(f"Processing {config['hf_path']}...")
        
        for i in range(min(len(ds), len(transcripts))):
            example = ds[i]
            t_entry = transcripts[i]
            
            audio_array = example['context']['audio']['array']
            sample_rate = example['context']['audio']['sampling_rate']
            
            wav_filename = f"sample_{i}.wav"
            wav_path = os.path.join(config["output_dir"], wav_filename)
            sf.write(wav_path, audio_array, sample_rate)
            
            words = []
            alignments = []

            for ts in t_entry.get('timestamps', []):

                word = str(ts.get('text', '')).strip()

                if word == "":
                    continue

                start_time = float(ts['start'])
                end_time = float(ts['end'])

                # remove broken timestamps
                if end_time <= start_time:
                    continue

                words.append(word)
                alignments.append(end_time)

            # fallback if ASR failed
            if len(words) == 0:
                words = ["speech"]
                alignments = [1.0]

            meta = example['other_attributes']['meta_psudo']

            diarization = meta.get('speaker_diarization', [])

            if diarization and len(diarization) > 0:
                speaker_id = diarization[0].get('speaker', 'unknown')
            else:
                speaker_id = 'unknown'

            manifest_entries.append({
                "audio_filepath": os.path.abspath(wav_path),
                "duration": float(len(audio_array) / sample_rate),
                "text": t_entry.get('text', ''),
                "speaker_id": speaker_id,
                "words": words,
                "alignments": alignments
            })



    with open(FINAL_MANIFEST, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"SUCCESS: Extracted audio and created manifest with {len(manifest_entries)} entries.")

if __name__ == "__main__":
    process()