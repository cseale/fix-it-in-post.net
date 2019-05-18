import numpy as np
import scipy
from numpy import array, where, median, abs

from pesq_lib.pypesq import pypesq
from utils.synthesis_util import *


class Metrics:
    # used to get the spectral flux out of the stft of the signal (especially it is using the magnitude of the spectrum)
    @staticmethod
    def spectral_flux(magnitude_spectrum, sample_rate):
        # convert to frequency domain
        timebins, freqbins = np.shape(magnitude_spectrum)

        # when do these blocks begin (time in seconds)?
        timestamps = (np.arange(0, timebins - 1) * (timebins / float(sample_rate)))

        sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum)) ** 2, axis=1)) / freqbins

        return sf[1:], np.asarray(timestamps)

    # SNR using the beta sigma approximation, takes as input the spectral flux of the signal
    # Our current first choice for calculating Signal to Noise Ratio (SNR)
    @staticmethod
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

    @staticmethod
    def snr_mean_std_based(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return 10 * np.log10(np.sum(m / sd))

    @staticmethod
    def aggregate_metric_check(audio_files, model, limit=-1, window=scipy.signal.hamming(256, "periodic")):
        snr_noise = []
        snr_clean = []
        snr_denoised = []
        pesq_noise = []
        pesq_clean = []
        pesq_denoised = []

        if limit == -1 or limit > len(audio_files):
            limit = len(audio_files)

        for i in range(limit):
            # obtain clean and noisy samples
            y_clean, sr = get_audio(audio_id=i, audio_files=audio_files)
            y_noise, sr = get_noisy_audio(audio_id=i, audio_files=audio_files)

            length = len(y_noise)

            # Obtain sample
            win = window
            magnitude, phase = audio_to_sttft(y_noise, win)
            predictors = get_predictors(magnitude)
            predictors = np.array(predictors)
            sample = obtain_sample(predictors)

            # denoise sample
            audio_rec = denoise_audio(model, sample, phase, window, length)

            # calculate sttfts
            magnitude_clean, phase_clean = audio_to_sttft(y_clean, win)
            magnitude_noise, phase_noise = audio_to_sttft(y_noise, win)
            magnitude_denoised, phase_denoised = audio_to_sttft(audio_rec, win)

            # calculate SNR
            snr_clean.append(10 * np.log10(Metrics.DER_SNR(Metrics.spectral_flux(magnitude_clean, sr))))
            snr_noise.append(10 * np.log10(Metrics.DER_SNR(Metrics.spectral_flux(magnitude_noise, sr))))
            snr_denoised.append(10 * np.log10(Metrics.DER_SNR(Metrics.spectral_flux(magnitude_denoised, sr))))

            # calculate PESQ
            pesq_clean.append(pypesq(sr, y_clean, y_clean, 'nb'))
            pesq_noise.append(pypesq(sr, y_clean, y_noise, 'nb'))
            pesq_denoised.append(pypesq(sr, y_clean, audio_rec, 'nb'))

        metrics = {"snr": {"clean": np.mean(snr_clean), "noise": np.mean(snr_noise), "denoised": np.mean(snr_denoised)},
                   "pesq": {"clean": np.mean(pesq_clean), "noise": np.mean(pesq_noise),
                            "denoised": np.mean(pesq_denoised)}}
        return metrics

