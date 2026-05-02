

import librosa
import numpy as np
import scipy.signal

def get_loudness_features(y):
    
    rms = librosa.feature.rms(y=y)[0] #rms to get the average
    rms_nonzero = rms[rms > 0] # remove zeros to avoid errors

    if len(rms_nonzero) == 0:
        return {"loud_max": 0, "loud_min": 0, "loud_std": 0, "loud_range": 0}

    db = librosa.amplitude_to_db(rms_nonzero) # convert to decibels 
    loud_range = np.max(db) - np.min(db)

    if loud_range > 40:
        loudness_label = "Extreme: Shout/Whisper"
    else:
        loudness_label = "Normal"
        
    return {
        "loudness_label" : loudness_label,
        "loud_std": round(float(np.std(db)), 2),
        "loud_range": round(float(np.max(db) - np.min(db)), 2)
    }


def get_pitch_features(y, sr):
    
    #f0 represnts frequency which is the pitch
    f0 = librosa.yin(y=y, fmin=65, fmax=2093, sr=sr)
    voiced_f0 = f0[f0 > librosa.note_to_hz('C2')] #only the part with the voice

    f0 = f0[f0 > 0]
    if len(f0) == 0:
        return {"pitch_max": 0, "pitch_min": 0, "pitch_std": 0, "tremor_index": 0}

    
    # this part is for the tremor: measures how much the pitch 'shakes' from frame to frame
    voiced_f0 = f0[f0 > librosa.note_to_hz('C2')]
    smoothed_f0 = scipy.signal.medfilt(voiced_f0, kernel_size=3)
    
    differences = np.abs(np.diff(smoothed_f0))
    #tremor = np.mean(differences) / np.mean(voiced_f0)
    tremor = np.median(differences) / np.median(smoothed_f0)

    # using percentiles instead of min-max
    p95 = np.percentile(f0, 95)
    p05 = np.percentile(f0, 5)
    
    stable_range = p95 - p05
    std_dev = np.std(f0)
    

    if tremor > 0.05:
        tremor_label = "Shaky voice"
    else:
        tremor_label = "Stable voice"
    
    
    return {
        "pitch_range": stable_range,
        "pitch_std": std_dev,
        "pitch_label": "High" if std_dev > 60 else "Normal",

        "tremor_index": round(float(tremor), 4),
        "tremor_label": tremor_label 
    }


# this function is for measuring the rhythm which is the dragging of words and also 
# the stability (singing the words)
def get_rhythm_features(y, sr):

    # onset strength = likelihood and intensity of a sound occurring at a specific time
    onset_env = librosa.onset.onset_strength(y=y, sr=sr) #onset = where the word starts
    
    # tempo in bmp
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # clarity
    # more clarity = rhythmic/sing-song and lower clarity = irregular/stuttering
    # pulse_clarity = np.mean(librosa.feature.fourier_tempogram(onset_envelope=onset_env, sr=sr))

    # rhythmic variability (sd of Onset Strength)
    # variability = 'dragging' words
    rhythm_std = np.std(onset_env)


    # #making it human-readable: 
    # # clarity / stability
    # if pulse_clarity > 0.1:
    #     clarity_label = "High: Rhythmic"
    # elif pulse_clarity > 0.03:
    #     clarity_label = "Normal"
    # else:
    #     clarity_label = "Low: Stuttering/Irregular"


    # variability
    if rhythm_std > 3:
        rhythm_label = "High: Dragged words"
    elif rhythm_std > 1.5:
        rhythm_label = "Normal"
    else:
        rhythm_label = "Flat: Monotone"



    return {
        "tempo_bpm": round(float(tempo), 2),
        # "rhythm_stability": round(float(pulse_clarity), 4),
        # "rhythm_clarity_label": clarity_label,
        "onset_variance": round(float(rhythm_std), 2),
        "rhythm_variability_label": rhythm_label
    }




def get_final_tier_s(pitch_stats, loud_stats, rhythm_stats, text_info):
   
    # Tremor: distress and excitement
    #has_tremor = pitch_stats['tremor_label'] == "Shaky Voice"
    has_tremor = "shaky" in pitch_stats['tremor_label'].lower()

    # # Rhythm
    # is_clari_extreme = rhythm_stats['rhythm_clarity_label'] in ["High: Rhythmic", "Low: Stuttering/Irregular"]
    
    # # variability
    # is_vari_good = rhythm_stats['rhythm_variability_label'] in ["High: Dragged words", "Normal"]

    has_pitch_variation = pitch_stats.get('pitch_std', 0) > 80 # std of pitch is considered high/expressive when > 50Hz for speech
    has_loudness_variation = loud_stats.get('loud_std', 0) > 7 # std of loudness is considered high when >= 5 LU showing "spikes" in volume
    
    # range
    has_large_pitch_range = pitch_stats.get('pitch_range', 0) > 150
    has_large_loud_range = loud_stats.get('loud_range', 0) > 40

    # -- the distinguisher

    if has_tremor or loud_stats.get('loud_range', 0) > 45:
        return "High Intensity"
    
    # pitch AND volume both vary to be considered "High Intensity"
    if has_pitch_variation and has_loudness_variation:
        return "High Intensity"
    
    # rhythm (variability) 
    if rhythm_stats['rhythm_variability_label'] == "High: Dragged words":
        return "High Intensity"
    
    # big range of loudness
    if has_large_loud_range:
        return "High Intensity"
    
    # significant pitch movement
    if has_large_pitch_range:
        return "High Intensity"

    return "Neutral"


    # #has tremor then is considered high intensity
    # if has_tremor:
    #     return "High Intensity"
    
    # # has both, extreme clarity and good variability
    # if is_clari_extreme and is_vari_good:
    #     return "High Intensity"
    
    # # large variations in both pitch AND loudness
    # if has_pitch_variation or has_loudness_variation:
    #     return "High Intensity"
    
    # # big range of loudness
    # if has_large_loud_range:
    #     return "High Intensity"
    
    # # significant pitch movement
    # if has_large_pitch_range:
    #     return "High Intensity"
    
    # # everything else is neutral
    # return "Neutral"

