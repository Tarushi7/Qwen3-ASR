
# # TO GET OVERALL SUMMARY TABLE
# import json
# import os
# import re
# from tqdm import tqdm

# RESULTS_DIR = "qwen_instruct_results"

# analysis_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_qwen_analysis.json")]
# analysis_files.sort()

# print("\n" + "="*90)
# print(f"{'Dataset File':<40} | {'Total':<6} | {'Keep':<6} | {'Discard':<8} | {'Err':<4} | {'Keep %':<6}")
# print("-" * 90)

# grand_total = 0
# grand_kept = 0
# grand_discarded = 0
# grand_errors = 0

# for filename in analysis_files:
#     file_path = os.path.join(RESULTS_DIR, filename)
    
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"Error reading {filename}: {e}")
#         continue

#     kept_count = 0
#     discarded_count = 0
#     error_count = 0

#     for entry in data:
#         results = entry.get("instruct_model_results", {})
#         raw_response = results.get("full_response", "")
        
#         if not raw_response:
#             error_count += 1
#             continue

#         try:
#             json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            
#             if json_match:
#                 clean_json = json_match.group(0)
#             else:
#                 clean_json = raw_response.strip()

#             clean_json = clean_json.replace("```json", "").replace("```", "").strip()
            
#             if clean_json.count('{') > clean_json.count('}'):
#                 clean_json += '}'
            
#             parsed_response = json.loads(clean_json)
            
#             verdict = str(parsed_response.get("Verdict", "")).upper()
            
#             if "KEEP" in verdict:
#                 kept_count += 1
#             elif "DISCARD" in verdict:
#                 discarded_count += 1
#             else:
#                 upper_raw = raw_response.upper()
#                 if '"VERDICT": "KEEP"' in upper_raw or '"VERDICT":"KEEP"' in upper_raw:
#                     kept_count += 1
#                 elif '"VERDICT": "DISCARD"' in upper_raw or '"VERDICT":"DISCARD"' in upper_raw:
#                     discarded_count += 1
#                 else:
#                     error_count += 1
                
#         except Exception:
#             upper_raw = raw_response.upper()
#             if '"VERDICT": "KEEP"' in upper_raw or '"VERDICT":"KEEP"' in upper_raw:
#                 kept_count += 1
#             elif '"VERDICT": "DISCARD"' in upper_raw or '"VERDICT":"DISCARD"' in upper_raw:
#                 discarded_count += 1
#             else:
#                 error_count += 1

#     total = len(data)
#     keep_rate = (kept_count / total) * 100 if total > 0 else 0
    
#     grand_total += total
#     grand_kept += kept_count
#     grand_discarded += discarded_count
#     grand_errors += error_count

#     print(f"{filename:<40} | {total:<6} | {kept_count:<6} | {discarded_count:<8} | {error_count:<4} | {keep_rate:>6.1f}%")

# # Overall Summary
# overall_rate = (grand_kept / grand_total) * 100 if grand_total > 0 else 0
# print("-" * 90)
# print(f"{'OVERALL TOTAL':<40} | {grand_total:<6} | {grand_kept:<6} | {grand_discarded:<8} | {grand_errors:<4} | {overall_rate:>6.1f}%")
# print("="*90 + "\n")






# TO GET DISCARD INDEXES
import json
import os
import re

RESULTS_DIR = "qwen_instruct_results"

analysis_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_qwen_analysis.json")]
analysis_files.sort()

def get_verdict(raw_response):
    if not raw_response:
        return "ERROR"
    
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0).replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            return str(parsed.get("Verdict", "")).upper()
    except:
        pass

    upper_raw = raw_response.upper()
    if '"VERDICT": "KEEP"' in upper_raw or '"VERDICT":"KEEP"' in upper_raw:
        return "KEEP"
    if '"VERDICT": "DISCARD"' in upper_raw or '"VERDICT":"DISCARD"' in upper_raw:
        return "DISCARD"
    
    return "UNKNOWN"

print("\n" + "="*60)
print("DISCARDED INDEXES PER DATASET")
print("="*60)

for filename in analysis_files:
    file_path = os.path.join(RESULTS_DIR, filename)
    lang_name = filename.replace("_qwen_analysis.json", "").upper()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    discarded_indexes = []

    for entry in data:
        results = entry.get("instruct_model_results", {})
        raw_response = results.get("full_response", "")
        
        verdict = get_verdict(raw_response)
        
        if "DISCARD" in verdict:
            discarded_indexes.append(str(entry.get("index")))

    print(f"\n>>> {lang_name} ({len(discarded_indexes)} items discarded):")
    
    if discarded_indexes:
        wrapped_list = ", ".join(discarded_indexes)
        print(wrapped_list)
    else:
        print("None.")

print("\n" + "="*60)
print("Scan Complete.")



