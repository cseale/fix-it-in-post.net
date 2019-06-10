import os
import librosa
import scipy
import numpy as np
import progressbar
import pickle
import sys

ALL_FILES = 8192
MINI_FILES = 512

PADDING_VALUE = -1

clean_audio_dir = "clean_trainset_28spk_wav/"
raw_dir = "./data/raw/edinburgh-noisy-speech-db/"


def get_stft(y, orig_sr, window_length, overlap, target_sr):
    # define vars
    window_length = window_length
    win = scipy.signal.hamming(window_length, "periodic")
    overlap = round(overlap * window_length)
    fft_length = window_length
    # downsampling
    target_sr = target_sr
    y = librosa.resample(y, target_sr=target_sr, orig_sr=orig_sr)
    sr = target_sr
    # padding, because input must be multiple of fft window
    n = len(y)
    y_pad = librosa.util.fix_length(y, n + fft_length // 2)
    # get STFT
    D = librosa.stft(y_pad.astype(np.float32),
                     n_fft=fft_length,
                     win_length=window_length,
                     window=win,
                     hop_length=overlap)
    return D


def get_directory_name(window_length, overlap, sampling_rate, num_segments):
    name = "RNN_wtf" + "w" + str(window_length) + "o" + str(overlap) + "sr" + str(sampling_rate) + "n" + str(
        num_segments) + "/"
    name = name.replace("0.", "").replace(".0", "")
    return name


def get_longest_audio_name(audio_files, window_length=256, overlap=0.75, sampling_rate=8e3):
    audio_files_duration = [librosa.get_duration(filename=get_audio_file_path(elem)) for elem in audio_files]
    audio_name = audio_files[np.argmax(audio_files_duration)]
    y, sr = librosa.load(get_audio_file_path(audio_name))
    D = get_stft(y, sr, window_length, overlap, sampling_rate)
    magnitude_noise = np.abs(D)
    return magnitude_noise.shape


def get_audio_file_path(audio_file_name):
    return raw_dir + clean_audio_dir + audio_file_name


def limit_or_pad(magnitude, final_length):
    if final_length < magnitude.shape[1]:
        return magnitude[:, :final_length]
    to_ret = np.zeros((magnitude.shape[0], final_length))
    to_ret[:, :magnitude.shape[1]] = magnitude
    return to_ret


def process_audio(process_all=False, window_length=256, overlap=0.75, sampling_rate=8e3, num_segments=6):
    print("Processing Audio...")
    print("window_length = " + str(window_length))
    print("overlap = " + str(overlap))
    print("sampling_rate = " + str(sampling_rate))
    print("num_segments = " + str(num_segments))

    directory_name = get_directory_name(window_length, overlap, sampling_rate, num_segments)
    processed_dir = "./data/processed/edinburgh-noisy-speech-db/" + directory_name
    processed_filename = "train.pkl"
    log_trainset = "log_trainset_28spk.txt"
    audio_files = []

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

    max_lenght = get_longest_audio_name(audio_files)

    print("Max Length:" + str(max_lenght))

    with progressbar.ProgressBar(max_value=len(audio_files)) as bar:
        file_index = 0
        for i, f in enumerate(audio_files):

            # load file
            y, sr = librosa.load(get_audio_file_path(f))
            D = get_stft(y, sr, window_length, overlap, sampling_rate)
            D = np.abs(D)

            magnitude = limit_or_pad(D, num_segments * 10)

            # create noisy version
            noise_amp = 0.05 * np.amax(y)
            y_noise = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])

            D_noise = get_stft(y_noise, sr, window_length, overlap, sampling_rate)
            D_noise = np.abs(D_noise)
            magnitude_noise = limit_or_pad(D_noise, num_segments * 10)
            for i in range(10):
                dataset = {
                    "predictors": None,
                    "targets": None
                }
                dataset["predictors"] = magnitude_noise[:, num_segments * i:num_segments * (i + 1)]
                dataset["targets"] = magnitude[:, num_segments * i:num_segments * (i + 1)]

                with open(processed_dir + "sample." + str(file_index) + ".pkl", 'wb') as handle:
                    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                file_index = file_index + 1

            bar.update(i)

        # write number of files to info file
        f = open(processed_dir + "info", "w+")
        f.write(str(file_index - 1))
        f.close()
    print("processing finished")

    return dataset, processed_dir, directory_name


if __name__ == "__main__":
    process_all = False
    upload = False
    try:
        process_all = sys.argv[1] == "all"
    except:
        process_all = False

    dataset, processed_dir, key = process_audio(process_all=process_all)
