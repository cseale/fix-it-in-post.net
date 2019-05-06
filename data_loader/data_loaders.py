from base import BaseDataLoader
import torch
import pickle
import boto3
import getpass
from torch.utils.data.dataset import Dataset  # For custom datasets

class STFTDataLoader(BaseDataLoader):
    """
    STFT data loading demo using BaseDataLoader

    Load preprocessed data
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = EdinburghDataset(self.data_dir)

        super(STFTDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class EdinburghDataset(Dataset):

    def __init__(self, data_dir): 
        if getpass.getuser() == "ec2-user":
            d = downloadS3File()
        else:
            d = useLocalFile()

        # TODO: use transformer
        self.labels = torch.from_numpy(d["targets"])
        self.data = torch.from_numpy(d["predictors"])
        self.data = self.data.view(self.data.shape[0], -1)
        print(self.data.shape[0])
        print("Label dimensions" + str(self.labels.shape))
        print("Data dimensions" + str(self.data.shape))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def useLocalFile():
    data_dir = "./data/processed/edinburgh-noisy-speech-db/train.pkl"
    with open(data_dir, 'rb') as f:
        d = pickle.load(f)
    return d

def downloadS3File():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('fix-it-in-post')
    obj = bucket.Object('train.128')
    d = obj.get().get('Body').read()
    d = pickle.loads(d)
    return d