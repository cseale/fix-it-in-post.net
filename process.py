import os
import librosa
import scipy
import numpy as np
import progressbar
import pickle

def get_stft(y, sr):
    # define vars
    window_length = 256;
    win = scipy.signal.hamming(window_length,"periodic");
    overlap = round(0.75 * window_length);
    fft_length = window_length;
    # downsampling
    target_sr = 8e3;
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
    raw_dir = "../data/raw/edinburgh-noisy-speech-db/"
    processed_dir = "../data/processed/edinburgh-noisy-speech-db/"
    clean_audio_dir = "clean_trainset_28spk_wav/"
    noisy_audio_dir = "noisy_trainset_28spk_wav/"
    log_trainset = "log_trainset_28spk.txt"
    audio_files = []
    
    num_features  = 129;
    num_segments  = 8;

    dataset = {
        "predictors": [],
        "targets": []
    }
    
    # list files
    f = open(raw_dir + log_trainset, "r")
    for x in f:
        audio_files.append(x.split()[0] + ".wav")
    f.close()
    
    if process_all == False:
        audio_files = audio_files[0:128]
        
    print("Processing " + str(len(audio_files)) + " files")
    
    with progressbar.ProgressBar(max_value=len(audio_files)) as bar:
        for i, f in enumerate(audio_files):
            for a in [clean_audio_dir, noisy_audio_dir]:
                y, sr = librosa.load(raw_dir + a + f)
                D = get_stft(y, sr)
                magnitude = np.abs(D)
                
                for segment_index in range(magnitude.shape[1] - num_segments):
                    if a == noisy_audio_dir:
                        dataset["predictors"].append(magnitude[:, segment_index:segment_index + num_segments])
                    else:
                        dataset["targets"].append(magnitude[:,segment_index + num_segments])
            bar.update(i)
            
    with open(processed_dir + "train.pkl", 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("processing finished")        
    
    return dataset
            
processed_dataset = process_audio()