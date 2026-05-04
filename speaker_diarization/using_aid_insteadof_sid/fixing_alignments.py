import json
import os

INPUT_MANIFEST = "/home/tarushi/tarushi_folder/speaker_diarization/input_manifest_aid.json"
OUTPUT_MANIFEST = "/home/tarushi/tarushi_folder/speaker_diarization/input_manifest_with_alignments.json"

def fix_manifest():
    fixed_count = 0
    skipped_count = 0

    with open(INPUT_MANIFEST, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_MANIFEST, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            item = json.loads(line)
            duration = float(item.get('duration', 0.0))
            
            if duration <= 0.01:
                skipped_count += 1
                continue

            new_alignments = []
            source_words = item.get('words', [])
            
            if source_words:
                for w in source_words:
                    start = float(w.get('start_time', 0.0))
                    end = float(w.get('end_time', duration))
                    word_val = w.get('word').strip() if w.get('word') else "speech"
                    
                    if end > start:
                        new_alignments.append([start, end, word_val])
            
            if len(new_alignments) < 2:
                mid = duration / 2
                new_alignments = [
                    [0.0, round(mid, 3), "speech_part1"],
                    [round(mid, 3), round(duration, 3), "speech_part2"]
                ]

            item['alignments'] = new_alignments
            item['duration'] = duration
            item['text'] = " ".join([str(a[2]) for a in new_alignments])
            
            f_out.write(json.dumps(item) + '\n')
            fixed_count += 1

    print(f"Done! Processed: {fixed_count} | Skipped: {skipped_count}")
    print(f"USE THIS PATH IN YOUR RUN SCRIPT: {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    fix_manifest()



