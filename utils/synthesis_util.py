import librosa.display
import librosa.util
import numpy as np
import torch

clean_audio_dir = "./data/raw/edinburgh-noisy-speech-db/clean_trainset_28spk_wav/"
raw_dir = "./data/raw/edinburgh-noisy-speech-db/"
log_trainset = "log_trainset_28spk.txt"


def load_files():
    audio_files = []
    # list files
    f = open(raw_dir + log_trainset, "r")
    for x in f:
        audio_files.append(x.split()[0] + ".wav")
    f.close()

    return audio_files


def get_clean_audio_file(audio_id, audio_files):
    audio_file = audio_files[audio_id]
    clean_audio_f = clean_audio_dir + audio_file
    return clean_audio_f


def get_audio(audio_id, audio_files, resample=True):
    clean_audio_f = get_clean_audio_file(audio_id, audio_files)
    y, sr = librosa.load(clean_audio_f)

    if resample:
        y, sr = resample_to_8k(y, sr)

    return y, sr


def transform_to_noisy(y):
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.15 * np.random.uniform() * np.amax(y)
    y_noise = y.astype('float64') + noise_amp * np.random.normal(size=y.shape[0])

    return y_noise


def get_noisy_audio(audio_id, audio_files):
    y_noise, sr = get_audio(audio_id, audio_files)
    y_noise = transform_to_noisy(y_noise)
    return y_noise, sr


def resample_to_8k(audio, sr):
    fs = 8e3
    return librosa.resample(audio, target_sr=fs, orig_sr=sr), fs


def audio_to_sttft(y_audio, win, hop_length=64, window_length=256):
    fft_length = window_length

    n = len(y_audio)
    y_padded = librosa.util.fix_length(y_audio, n + fft_length // 2)
    D = librosa.stft(y_padded.astype(np.float32),
                     n_fft=fft_length,
                     win_length=window_length,
                     window=win,
                     hop_length=hop_length)

    magnitude, phase = librosa.magphase(D)
    return magnitude, phase


def get_predictors(magnitude, num_segments=8):
    predictors = []
    for segment_index in range(magnitude.shape[1] - num_segments + 1):
        predictors.append(magnitude[:, segment_index:segment_index + num_segments])

    return predictors


def denoise_audio(model, sample, phase, window, length, num_segments=8, window_length=256, hop_length=64):
    y_pred = model(sample)
    y_pred = y_pred.detach().numpy().transpose()
    D_rec = y_pred * phase[:, num_segments - 1:]
    audio_rec = librosa.istft(D_rec,
                              length=length,
                              win_length=window_length,
                              window=window,
                              hop_length=hop_length)
    return audio_rec


def obtain_sample(predictors):
    predictors = np.array(predictors)
    sample = torch.from_numpy(predictors)
    sample = sample.view(sample.shape[0], -1)
    return sample
