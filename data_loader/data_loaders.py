from base import BaseDataLoader
import torch
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

        # TODO: use transformer
        self.labels = torch.from_numpy(d["targets"])
        self.data = torch.from_numpy(d["predictors"])
        self.data = self.data.view(self.data.shape[0], -1)
        print("Label dimensions" + str(self.labels.shape))
        print("Data dimensions" + str(self.data.shape))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
