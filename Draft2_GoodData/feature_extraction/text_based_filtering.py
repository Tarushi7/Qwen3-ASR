
from collections import Counter
import string
import re

#specific to language, based on meralion transcriptions
def get_text_metrics(text, timestamps, y, sr, lang_code):
    true_duration = len(y) / sr
    total_spoken_time = sum(ts[1] - ts[0] for ts in timestamps)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()

    if lang_code == "zh":
        # Chinese: Count every individual character as a "word"
        # Includes standard and extended CJK Unified Ideographs
        tokens = re.findall(r'[\u4e00-\u9fff]', text)
    elif lang_code == "ta":
        # Tamil: Capture Tamil Unicode blocks + English (for code-switching)
        tokens = re.findall(r'[\u0b80-\u0bff]+|[a-zA-Z0-9]+', text)
    else:
        # English/Malay: Standard whitespace/alphanumeric splitting
        tokens = re.findall(r'[a-zA-Z0-9]+', text)

    words = [t.lower() for t in tokens]
    word_count = len(words)
    
    # Calculate WPM based on the specific tokenization
    safe_dur = max(true_duration, 0.001)
    wpm = (word_count / safe_dur) * 60
    
    return {
        "true_duration": true_duration,
        "talk_time": total_spoken_time,
        "speech_ratio": total_spoken_time / safe_dur,
        "wpm": wpm,
        "word_count": word_count,
        "words": words 
    } 

from collections import Counter

def get_stutter_stats(words, window_size=5):
    if not words or len(words) < window_size:
        return {"word_repetition": 0}
    
    total_words = len(words)
    stutter_windows = 0
    
    # sliding window
    for i in range(len(words) - window_size + 1):
        window = words[i : i + window_size]
        window_counts = Counter(window)
        
        # word repetition
        if any(count >= 3 for count in window_counts.values()):
            stutter_windows += 1

    # ratio of windows that have word repetitions``
    word_repetition = stutter_windows / (total_words - window_size + 1)
    
    return {
        "word_repetition": round(word_repetition, 4),
        "total_words": total_words
    }



def get_redundancy_score(words, n=3):
    #calculates repeat for 3-word phrases instead of word repetition
    # Now takes 'words' list instead of string to capture chinese characters as individual tokens
    if not words or len(words) < n:
        return 0.0
    
    # Create n-grams from the word list
    grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    total_grams = len(grams)
    unique_grams = len(set(grams))
    
    # so if 90% of the phrases are repeats, redundancy will be ~0.9
    redundancy = (total_grams - unique_grams) / total_grams
    return round(redundancy, 4)




def get_final_tier(text_info, stutter_info, redundancy_score, lang_code, stutter_threshold):
    # minimum WPM specific to language
    # Chinese --> more characters, Tamil --> fewer words
    min_wpm = {"en": 70, "ms": 70, "zh": 110, "ta": 50}.get(lang_code, 70)

    if text_info['wpm'] < min_wpm:
        return "Low Quality: Discard"

    if text_info['speech_ratio'] < 0.50: 
        return "Low Quality: Discard"
    
    if stutter_info['word_repetition'] > stutter_threshold:
        return "Low Quality: Discard"

    if redundancy_score > 0.4:
        return "Low Quality: Discard"

    return "High Quality"




