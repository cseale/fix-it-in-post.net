from base import BaseDataLoader
import torch
import pickle
import boto3
import botocore
import os
import getpass
from process import process_audio, get_directory_name 
from torch.utils.data.dataset import Dataset  # For custom datasets

EDINBURGH_DATA_DIR = "./data/processed/edinburgh-noisy-speech-db/"

class STFTDataLoader(BaseDataLoader):
    """
    STFT data loading demo using BaseDataLoader

    Load preprocessed data
    """
    def __init__(self, batch_size, shuffle=False, validation_split=0.0, num_workers=1, window_length = 256, overlap = 0.75, sampling_rate = 8e3, num_segments = 8):
        self.dataset = EdinburghDataset(window_length, overlap, sampling_rate, num_segments)
        super(STFTDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class EdinburghDataset(Dataset):

    def __init__(self, window_length, overlap, sampling_rate, num_segments): 
        self.data_dir = get_directory_name(window_length, overlap, sampling_rate, num_segments)
        # check if dir exists
        self.use_s3 = False
        if checkIfDataExistsOnLocal(self.data_dir):
            print("data is already processed...")
        elif checkIfDataExistsOnS3(self.data_dir) or getpass.getuser() == "ec2-user":
            self.use_s3 = True
            print("data is on S3")
        else:
            print("no data, processing locally...")
            process_audio(process_all=False, window_length = window_length, overlap = overlap, sampling_rate = sampling_rate, num_segments = num_segments)

        self.length = readLengthFile(self.data_dir, self.use_s3)
        num_features = (window_length/2) + 1
        self.data = torch.zeros([self.length, num_features * num_segments])
        self.labels = torch.zeros([self.length, num_features])

    def __getitem__(self, index):
        # TODO: should check for zeros
        d = readSampleFile(self.data_dir, index, self.use_s3)
        self.data[index] = torch.from_numpy(d['predictors']).reshape(1, -1)
        self.labels[index] = torch.from_numpy(d['targets'])

        print(self.data[index])
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def readLengthFile(data_dir, use_s3):
    info_filename = "info"
    if use_s3:
        return int(downloadS3File(data_dir, info_filename))
    else:
        return int(readLocalFile(data_dir, info_filename))

def readSampleFile(data_dir, index, use_s3):
    filename = "sample." + str(index) + ".pkl"   
    if use_s3:
        return pickle.loads(downloadS3File(data_dir, filename))
    else:
        return pickle.loads(readLocalFile(data_dir, filename))

def readLocalFile(data_dir, filename):
    data_dir = EDINBURGH_DATA_DIR + data_dir
    with open(data_dir + filename, 'rb') as f:
        d = f.read()
    return d

def downloadS3File(data_dir, filename):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('fix-it-in-post')
    obj = bucket.Object(data_dir + filename)
    d = obj.get().get('Body').read()
    return d

def checkIfDataExistsOnS3(data_dir):
    s3 = boto3.resource('s3')
    try:
        s3.Object('fix-it-in-post', data_dir + 'sample.1.pkl').load()
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False 

def checkIfDataExistsOnLocal(data_dir):
    data_dir = EDINBURGH_DATA_DIR + data_dir
    return os.path.exists(data_dir)
