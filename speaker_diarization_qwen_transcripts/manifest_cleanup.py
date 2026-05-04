
## to remove unaligned data

# import json

# input_manifest = 'processed_input_manifest.jsonl'
# cleaned_manifest = 'cleaned_input_manifest.jsonl'

# with open(input_manifest, 'r') as f_in, open(cleaned_manifest, 'w') as f_out:
#     for line in f_in:
#         data = json.loads(line)
#         # Check if alignments has at least 2 points
#         if len(data.get('alignments', [])) >= 2:
#             f_out.write(json.dumps(data) + "\n")





## to remove timestamps that don't match file length

# import json

# input_path = 'cleaned_input_manifest.jsonl'
# output_path = 'aligned_manifest.jsonl'
# removed_count = 0

# with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
#     for line in f_in:
#         data = json.loads(line)
#         duration = data.get('duration', 0)
#         aligns = data.get('alignments', [])
        
#         if aligns and max(aligns) <= duration:
#             f_out.write(json.dumps(data) + "\n")
#         else:
#             removed_count += 1

# print(f"Removed {removed_count} broken rows. Your new manifest is: {output_path}")




## to make each sample unique to each speaker erather than simply splitting each sample to multiple speakers

import json

input_path = 'aligned_manifest.jsonl'
output_path = 'final_manifest.jsonl'

with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
    for i, line in enumerate(f_in):
        data = json.loads(line)
        
        # Change the speaker_id to be unique for every single segment
        data['speaker_id'] = f"speaker_{i}"
        
        f_out.write(json.dumps(data) + "\n")

print(f"Done! Created {i+1} unique speakers in {output_path}")