import json
import os

def process_manifest_to_nemo(input_path, output_path, gap_threshold=1.5):
    new_entries = []
    
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            words = data.get('words', [])
            aligns = data.get('alignments', [])
            audio_path = data.get('audio_filepath', '')
            
            if not words or not aligns:
                continue

            current_speaker_idx = 0
            start_word_idx = 0
            
            
            for i in range(len(aligns) - 1):
                if (aligns[i+1] - aligns[i]) > gap_threshold:
                    segment_words = words[start_word_idx : i+1]
                    segment_aligns = aligns[start_word_idx : i+1]
                    
                    new_entries.append({
                        "audio_filepath": audio_path,
                        "duration": data['duration'],
                        "text": " ".join(segment_words),
                        "speaker_id": f"speaker_{current_speaker_idx}",
                        "words": segment_words,
                        "alignments": segment_aligns
                    })
                    
                    current_speaker_idx += 1
                    start_word_idx = i + 1
            
            final_words = words[start_word_idx:]
            final_aligns = aligns[start_word_idx:]
            new_entries.append({
                "audio_filepath": audio_path,
                "duration": data['duration'],
                "text": " ".join(final_words),
                "speaker_id": f"speaker_{current_speaker_idx}",
                "words": final_words,
                "alignments": final_aligns
            })

    
    with open(output_path, 'w') as f_out:
        for entry in new_entries:
            f_out.write(json.dumps(entry) + "\n")
            
    print(f"Processed {len(new_entries)} segments. New manifest saved to: {output_path}")


process_manifest_to_nemo('qwen_input_manifest.jsonl', 'processed_input_manifest.jsonl')