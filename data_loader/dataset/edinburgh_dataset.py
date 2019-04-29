import pickle

from torch.utils.data.dataset import Dataset  # For custom datasets


class EdinburghDataset(Dataset):
    dir_path = "data/raw/edinburgh-noisy-speech-db/"

    def __init__(self, file_name):
        with open(self.dir_path + file_name, 'rb') as f:
            d = pickle.load(f)

        self.labels = d["targets"]
        self.attributes = d["predictors"]

    def __getitem__(self, index):
        return self.attributes[index], self.labels[index]

    def __len__(self):
        return len(self.labels.index)
