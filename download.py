import sys
import time
import urllib.request
import os
import zipfile

DATA_DIR = "./data/raw/edinburgh-noisy-speech-db/"
FILENAME = "clean_trainset_28spk_wav"
FILE_URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y"
LOGFILES = "logfiles"
LOGFILES_URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/logfiles.zip?sequence=4&isAllowed=y"

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
    
def unzip(path, file):
    with zipfile.ZipFile(file,"r") as zip_ref:
        zip_ref.extractall(path)
        zip_ref.close()
        print("... data unzipped")
    os.remove(file)
    
# download audio and text files
print("creating directories...")
path = DATA_DIR + FILENAME + "/"
if not os.path.exists(DATA_DIR + FILENAME):
    os.makedirs(DATA_DIR + FILENAME);

print("downloading audio files...")
f = urllib.request.urlretrieve(FILE_URL, DATA_DIR + FILENAME + ".zip", reporthook)
print("...")
print("unzipping " + str(t[0]) + " to " + path)
unzip(path, f[0])

print("downloading log files...")
t = urllib.request.urlretrieve(LOGFILES_URL, DATA_DIR + LOGFILES + ".zip", reporthook)
print("...")
print("unzipping " + str(t[0]) + " to " + path)
unzip(DATA_DIR, t[0])

print("zip files deleted...")
print("download finished")
