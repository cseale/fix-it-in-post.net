from torchvision import datasets, transforms
from base import BaseDataLoader
import pickle
from torch.utils.data.dataset import Dataset  # For custom datasets
import os 

class STFTDataLoader(BaseDataLoader):
    """
    STFT data loading demo using BaseDataLoader

    Load preprocessed data
    """
    def __init__(self, batch_size, shuffle=False, validation_split=0.0, num_workers=1):
        self.file_name = "train.pkl"
        self.dataset = EdinburghDataset(self.file_name)

        super(STFTDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class EdinburghDataset(Dataset):
    print("We are here: " + os.path.dirname(os.path.realpath(__file__)))
    
    dir_path = "./data/processed/edinburgh-noisy-speech-db/"

    def __init__(self, file_name):
        with open(self.dir_path + file_name, 'rb') as f:
            d = pickle.load(f)

        self.labels = d["targets"]
        self.attributes = d["predictors"]
        
        print(self.attributes.shape)

    def __getitem__(self, index):
        return self.attributes[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
