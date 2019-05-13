from scipy.io import wavfile
from pesq_lib.pypesq import pypesq

rate, ref = wavfile.read("pesq_lib/audio/speech.wav")
rate, deg = wavfile.read("pesq_lib/audio/speech_bab_0dB.wav")

print(pypesq(rate, ref, deg, 'wb'))
print(pypesq(rate, ref, deg, 'nb'))