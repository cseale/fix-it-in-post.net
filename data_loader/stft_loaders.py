from base import BaseDataLoader
from data_loader.dataset.edinburgh_dataset import EdinburghDataset


class STFTDataLoader(BaseDataLoader):
    """
    STFT data loading demo using BaseDataLoader

    Load preprocessed data
    """
    def __init__(self, file_name, batch_size, shuffle=False, validation_split=0.0, num_workers=1):
        self.file_name = file_name
        self.dataset = EdinburghDataset(self.file_name)

        super(STFTDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
