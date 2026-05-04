# # checking speaker segments
# import json
# from collections import Counter

# speaker_counts = Counter()
# with open('cleaned_input_manifest.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         speaker_counts[data['speaker_id']] += 1

# print("Speaker Distribution:")
# for spk, count in speaker_counts.most_common():
#     print(f"ID: {spk} | Valid Segments: {count}")



# checking max alignment
import json

manifest_path = 'cleaned_input_manifest.jsonl'
flagged_entries = 0

with open(manifest_path, 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        duration = data.get('duration', 0)
        aligns = data.get('alignments', [])
        
        if not aligns:
            continue
            
        max_align = max(aligns)
        
        # Check if the alignment is physically impossible for the file
        if max_align > duration:
            print(f"Row {i} | File: {data['audio_filepath']}")
            print(f"  - Claimed Duration: {duration}s")
            print(f"  - Max Alignment: {max_align}s (OUT OF BOUNDS)")
            flagged_entries += 1

print(f"\nTotal problematic entries: {flagged_entries}")