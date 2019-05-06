import os
import librosa
import scipy
import numpy as np
import progressbar
import pickle
import sys


WINDOW_LENGTH = 1024
NUM_FEATURES = (WINDOW_LENGTH / 2) + 1
NUM_SEGMENTS = 8
OVERLAP = 0.75
SAMPLING_RATE = 8e3

ALL_FILES = 8192
MINI_FILES = 128

def get_stft(y, sr):
    # define vars
    window_length = WINDOW_LENGTH
    win = scipy.signal.hamming(window_length,"periodic")
    overlap = round(OVERLAP * window_length)
    fft_length = window_length
    # downsampling
    target_sr = SAMPLING_RATE
    y = librosa.resample(y, target_sr = target_sr, orig_sr = sr)
    sr = target_sr
    # padding, because input must be multiple of fft window
    n = len(y)
    y_pad = librosa.util.fix_length(y, n + fft_length // 2)
    # get STFT
    D = librosa.stft(y_pad.astype(np.float32),
        n_fft = fft_length,
        win_length = window_length,
        window = win,
        hop_length = overlap)
    return D

def process_audio(process_all = False):
    raw_dir = "./data/raw/edinburgh-noisy-speech-db/"
    processed_dir = "./data/processed/edinburgh-noisy-speech-db/" + "w" + str(WINDOW_LENGTH) + "o" + str(OVERLAP) + "sr" + str(SAMPLING_RATE) + "/"
    processed_dir = processed_dir.replace("0.", "").replace(".0", "")
    processed_filename = "train.pkl"
    clean_audio_dir = "clean_trainset_28spk_wav/"
    log_trainset = "log_trainset_28spk.txt"
    audio_files = []
    
    num_features  = NUM_FEATURES;
    num_segments  = NUM_SEGMENTS;
    
    # list files
    f = open(raw_dir + log_trainset, "r")
    for x in f:
        audio_files.append(x.split()[0] + ".wav")
    f.close()
    

    audio_files_count = ALL_FILES
    if process_all == False:
        audio_files_count = MINI_FILES
    audio_files = audio_files[0:audio_files_count]

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    print("Processing " + str(len(audio_files)) + " files")
    print("Storing in " + processed_dir)
    # write number of files to info file
    f= open(processed_dir + "info","w+")
    f.write(str(audio_files_count))
    f.close()
    
    with progressbar.ProgressBar(max_value=len(audio_files)) as bar:
        for i, f in enumerate(audio_files):
            dataset = {
                "predictors": [],
                "targets": []
            }
            for a in ["clean", "noisy"]:
                y, sr = librosa.load(raw_dir + clean_audio_dir + f)
                
                if a == "noisy":
                    noise_amp = 0.15*np.random.uniform()*np.amax(y)
                    y = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])
                
                D = get_stft(y, sr)
                magnitude = np.abs(D)
                
                for segment_index in range(magnitude.shape[1] - num_segments):
                    if a == "noisy":
                        dataset["predictors"].append(magnitude[:, segment_index:segment_index + num_segments])
                    else:
                        dataset["targets"].append(magnitude[:,segment_index + num_segments])

            dataset["predictors"] = np.array(dataset["predictors"])
            dataset["targets"] = np.array(dataset["targets"])

            with open(processed_dir + "sample." + str(i) + ".pkl", 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

            bar.update(i)
    
    print("processing finished")        
    
    return dataset
     
process_all = False
try:
    process_all = sys.argv[1] == "all"
except:
    process_all = False
    
process_audio(process_all = process_all)