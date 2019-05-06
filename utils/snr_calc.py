import librosa.display
import librosa.util
import matplotlib.pyplot as plt
from IPython.display import Audio
import os
import scipy
import numpy as np
from numpy import array, where, median, abs


# used to get the spectral flux out of the stft of the signal (especially it is using the magnitude of the spectrum)
def spectral_flux(magnitude_spectrum, sample_rate):
    # convert to frequency domain
    timebins, freqbins = np.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0, timebins - 1) * (timebins / float(sample_rate)))

    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum)) ** 2, axis=1)) / freqbins

    return sf[1:], np.asarray(timestamps)


# SNR using the beta sigma approximation, takes as input the spectral flux of the signal
def DER_SNR(flux):
    flux = array(flux)

    # Values that are exactly zero (padded) are skipped
    flux = array(flux[where(flux != 0.0)])
    n = len(flux)

    # For spectra shorter than this, no value can be returned
    if (n > 4):
        signal = median(flux)

        noise = 0.6052697 * median(abs(2.0 * flux[2:n - 2] - flux[0:n - 4] - flux[4:n]))

        return float(signal / noise)

    else:

        return 0.0


clean_audio_dir = "../data/raw/edinburgh-noisy-speech-db/clean_trainset_28spk_wav/"
audio_file = "p236_002.wav"
clean_audio_f = clean_audio_dir + audio_file
Audio(clean_audio_f)

noisy_audio_dir = "../data/raw/edinburgh-noisy-speech-db/noisy_trainset_28spk_wav/"
noisy_audio_f = noisy_audio_dir + audio_file

Audio(noisy_audio_f)

# Plot the waveforms, sr = 22050
y, sr = librosa.load(clean_audio_f)
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveplot(y, sr=sr)
plt.title('Clean Audio')

y_noise, sr_n = librosa.load(clean_audio_f)
plt.subplot(3, 1, 2)
librosa.display.waveplot(y_noise, sr=sr_n)
plt.title('Noisy Audio')
plt.tight_layout()

window_length = 256
win = scipy.signal.hamming(window_length,"periodic")
hop_length = round(0.25 * window_length)
fft_length = window_length

# downsampling to a new frequency fs
input_fs = sr
fs = 8e3

y = librosa.resample(y, target_sr = fs, orig_sr = sr)
sr = fs

n = len(y)

# pad the resampled audio to correctly process the STFT
y_pad = librosa.util.fix_length(y, n + fft_length // 2)

# obtain the STFT in terms of magnitude and phase in a matrix
D = librosa.stft(y_pad.astype(np.float32),
    n_fft = fft_length,
    win_length = window_length,
    window = win,
    hop_length = hop_length)
magnitude, phase = librosa.magphase(D)


print(f"SNR_db with function: {10*np.log10(DER_SNR(spectral_flux(magnitude, sr)))}")
print(D.shape)
# (number of frequency bands, number of window frames found along the signal)

librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

D_rec = magnitude * phase

audio_rec = librosa.istft(D_rec,
    length=n,
    win_length = window_length,
    window = win,
    hop_length = hop_length)

Audio(audio_rec, rate = sr)

num_features  = fft_length/2 + 1
num_segments  = 8