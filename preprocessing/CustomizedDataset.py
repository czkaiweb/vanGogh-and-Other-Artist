import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from preprocessing.ImageTranform import *

class CustomizedDataset(Dataset):
    def __init__(self, metaData, img_dir, transform=None, format = "jpg"):
        self.metaData = metaData
        self.img_dir = img_dir
        self.transform = transform
        self.suffix = '.'+format

    def setTransform(self,transform):
        self.transform = transform

    def __len__(self):
        return len(self.metaData)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                str(self.metaData["hash"].iloc[idx]) + self.suffix)
        image = Image.open(img_name)
        if image.mode == 'P':
            image = image.convert('RGB')
        artist = self.metaData["Artist"].iloc[idx]
        sample = {'image': image, 'artist': artist, 'hash':self.metaData["hash"].iloc[idx]}

        if self.transform:
            #print("Transforming image")
            sample['image'] = self.transform(image)

        return sample