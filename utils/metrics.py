import numpy as np
from numpy import array, where, median, abs


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
    #Our current first choice for calculating Signal to Noise Ratio (SNR)
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
        return 10*np.log10(np.sum(m/sd))
