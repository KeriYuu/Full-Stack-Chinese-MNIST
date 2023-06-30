import argparse
import pandas as pd
import os
from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
import sys
sys.path += ['/../../']
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
import text_recognizer.metadata.cmnist as metadata
from text_recognizer.stems.image import MNISTStem

class cMNISTDataset(Dataset):
    def __init__(self, img_folder, meta_df, transform=None):
        self.img_folder = img_folder
        self.meta_df = meta_df
        self.transform = transform

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_name = f'input_{row.suite_id}_{row.sample_id}_{row.code}.jpg'
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path)
        label = row.code - 1  # Assuming code ranges from 1 to 15
        if self.transform:
            img = self.transform(img)
        return img, label

class CMNIST(BaseDataModule):
    """cMNIST DataModule."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = os.path.join(metadata.DOWNLOADED_DATA_DIRNAME, 'data', 'data')
        self.transform =MNISTStem()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.mapping = metadata.MAPPING
        self.meta_file = os.path.join(metadata.DOWNLOADED_DATA_DIRNAME, 'chinese_mnist.csv')

    def prepare_data(self, *args, **kwargs) -> None:
        """Load meta data."""
        self.meta_df = pd.read_csv(self.meta_file)

    def setup(self, stage=None) -> None:
        """Create cMNISTDataset and split into train, val, test."""
        cmnist_full = cMNISTDataset(self.data_dir, self.meta_df, transform=self.transform)
        self.data_train, self.data_val = random_split(cmnist_full, [metadata.TRAIN_SIZE, metadata.VAL_SIZE])  # type: ignore
        self.data_test = cMNISTDataset(self.data_dir, self.meta_df, transform=self.transform)

if __name__ == "__main__":
    load_and_print_info(CMNIST)
