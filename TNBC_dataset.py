import copy
import os
import glob
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd


class TNBCDataset(Dataset):
    def __init__(self,
                 root: str,
                 crop_shape: tuple = None,
                 normalize: tuple = None,
                 random_hflip: bool = False,
                 random_brightness_contrast: bool = False,
                 mode: str = "train",
                 sample_img: bool = True,
                 fold=0,
                 multi_class=False,
                 transform=None,
                 cache=False
                 ):

        super(TNBCDataset, self).__init__()
        if 'oct' in root:
            self.phase_label_names = ['NOR', 'AMD', 'WAMD', 'DR', 'CSC', 'PED', 'MEM', 'FLD', 'EXU', 'CNV', 'RVO']
        elif 'TNBC' in root:
            self.phase_label_names = ['er', 'pr', 'her2', 'type_tn']
        else:
            raise Exception('Dataset not supported')

        self.num_phases_classes = len(self.phase_label_names)
        self.original_shape = (3, 256, 256)
        self.transform = transform
        assert os.path.isdir(root)
        self.root = root
        self.sample_list = []
        self.mode = mode
        self.sample_img = sample_img
        self.multi_class = multi_class
        self.crop_shape = crop_shape
        self.normalize = normalize
        self.random_hflip = random_hflip
        self.random_brightness_contrast = random_brightness_contrast
        self.dataset = self

        self.datapath = Path(self.root)

        self.dataset_path = self.datapath / f'{mode}_{fold}.csv'

        # Try to load data
        if not self.dataset_path.exists():
            raise Exception(f"{self.dataset_path} could not be found")

        self.dataset = pd.read_csv(self.dataset_path, index_col=0)

        # self.bin_column = bin_column
        self.label = self.get_mapped_labels()
        self.cache_dict = {}
        self.cache = cache
        if self.cache:
            for i in range(len(self.dataset)):
                img_path = self.dataset['filename'].iloc[i]
                img = cv2.imread(os.path.join(str(self.datapath), img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img_tensor = self.transform(img)
                self.cache_dict[i] = img_tensor


    def get_mapped_labels(self):
        if self.multi_class:
            return list(self.dataset[self.bin_column].values)
        else:
            if 'oct' in str(self.dataset_path):
                return list(
                    self.dataset[['NOR', 'AMD', 'WAMD', 'DR', 'CSC', 'PED', 'MEM', 'FLD', 'EXU', 'CNV', 'RVO']].values)
            elif 'TNBC' in str(self.dataset_path):
                return list(self.dataset[['er', 'pr', 'her2', 'type_tn']].values)
            else:
                raise Exception('Error in get mapped labels')

    def __getitem__(self, idx: int):
        img_path = self.dataset['filename'].iloc[idx]
        if self.cache:
            # print('Caching enabled'*10)
            if idx in self.cache_dict:
                img_tensor = self.cache_dict[idx]
            else:
                raise Exception('Cache no initalized properly')
        else:
            img = cv2.imread(os.path.join(str(self.datapath), img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = img
            if self.transform:
                img_tensor = self.transform(img)

        return img_tensor, torch.tensor(self.label[idx])

    def __len__(self):
        return len(self.label)

    def __dataset__(self):
        return self
