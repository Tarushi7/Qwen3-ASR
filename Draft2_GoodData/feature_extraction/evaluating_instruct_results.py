
import json
import os

# Define the results directory
RESULTS_DIR = "qwen_instruct_meralion_results"

# Get all instruction analysis files
analysis_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_instruct_analysis.json")]
analysis_files.sort()  # Sort to keep the output organized

print(f"{'Dataset File':<40} | {'Total':<6} | {'Keep':<6} | {'Discard':<8} | {'Err':<4} | {'Keep %':<6}")
print("-" * 85)

grand_total = 0
grand_kept = 0

for filename in analysis_files:
    file_path = os.path.join(RESULTS_DIR, filename)
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kept_count = 0
    discarded_count = 0
    error_count = 0

    for entry in data:
        # Get the response string stored by the instruction script
        results = entry.get("instruct_model_results", {})
        raw_response = results.get("full_response", "")
        
        try:
            # 1. Strip the "Neutral" wall if it exists (handles the loop error)
            # This cuts the string if it detects that massive repetition before parsing
            if "Neutral" in raw_response and len(raw_response) > 500:
                 # Attempt to find where the loop starts and truncate
                 # or simply try to find the "Verdict" keyword later
                 pass 

            # 2. Extract JSON using Regex (ignores text before/after the { })
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
            else:
                clean_json = raw_response.strip()

            # 3. Final sanitization
            clean_json = clean_json.replace("```json", "").replace("```", "").strip()
            
            # 4. Attempt to fix common "Cut-off" JSON errors manually
            if clean_json.count('{') > clean_json.count('}'):
                clean_json += '}' # Close a missing bracket
            
            parsed_response = json.loads(clean_json)
            
            # 5. Extract Verdict
            verdict = str(parsed_response.get("Verdict", "")).upper()
            
            if "KEEP" in verdict:
                kept_count += 1
            elif "DISCARD" in verdict:
                discarded_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            # If standard JSON fails, let's try a "Brute Force" string search
            # This is your safety net: if we see "KEEP" in the raw text, we count it.
            upper_raw = raw_response.upper()
            if '"VERDICT": "KEEP"' in upper_raw or '"VERDICT":"KEEP"' in upper_raw:
                kept_count += 1
            elif '"VERDICT": "DISCARD"' in upper_raw or '"VERDICT":"DISCARD"' in upper_raw:
                discarded_count += 1
            else:
                error_count += 1

    # Metrics for this specific file
    total = len(data)
    keep_rate = (kept_count / total) * 100 if total > 0 else 0
    
    # Update global totals
    grand_total += total
    grand_kept += kept_count

    print(f"{filename:<40} | {total:<6} | {kept_count:<6} | {discarded_count:<8} | {error_count:<4} | {keep_rate:>6.1f}%")

# Final Overall Summary
overall_rate = (grand_kept / grand_total) * 100 if grand_total > 0 else 0
print("-" * 85)
print(f"{'OVERALL TOTAL':<40} | {grand_total:<6} | {grand_kept:<6} | {'-':<8} | {'-':<4} | {overall_rate:>6.1f}%")




# #checking discarded indexes
# import os
# import re
# import json


# RESULTS_DIR = "qwen_instruct_meralion_results"

# analysis_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_instruct_analysis.json")]
# analysis_files.sort()

# print("--- DISCARDED INDEX REPORT ---")
# print("Target Directory:", RESULTS_DIR)
# print("-" * 30)

# for filename in analysis_files:
#     file_path = os.path.join(RESULTS_DIR, filename)
    
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     discarded_indices = []
#     error_indices = []

#     for entry in data:
#         results = entry.get("instruct_model_results", {})
#         raw_response = results.get("full_response", "")
#         idx = entry.get("index", "Unknown")
        
#         try:
            
#             json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
#             clean_json = json_match.group(0) if json_match else raw_response.strip()
            
#             clean_json = clean_json.replace("```json", "").replace("```", "").strip()
            
#             if clean_json.count('{') > clean_json.count('}'):
#                 clean_json += '}'
            
#             parsed_response = json.loads(clean_json)
#             verdict = str(parsed_response.get("Verdict", "")).upper()
            
#             if "DISCARD" in verdict:
#                 discarded_indices.append(idx)
                
#         except Exception:
#             upper_raw = raw_response.upper()
#             if '"VERDICT": "DISCARD"' in upper_raw or '"VERDICT":"DISCARD"' in upper_raw:
#                 discarded_indices.append(idx)
#             else:
#                 if '"VERDICT": "KEEP"' not in upper_raw and '"VERDICT":"KEEP"' not in upper_raw:
#                     error_indices.append(idx)

#     print(f"\nFILE: {filename}")
#     print(f"Total Discarded ({len(discarded_indices)}): {discarded_indices}")
#     if error_indices:
#         print(f"Unparseable/Errors ({len(error_indices)}): {error_indices}")
#     print("-" * 30)

# print("\nReport Complete.")


