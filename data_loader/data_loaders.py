from base import BaseDataLoader
import pickle
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
        with open(data_dir, 'rb') as f:
            d = pickle.load(f)

        self.labels = d["targets"]
        self.attributes = d["predictors"]

    def __getitem__(self, index):
        return self.attributes[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
