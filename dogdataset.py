import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class DogDataset(Dataset):
    def __init__(self, img_dir, transform1=None, transform2=None):

        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform1 = transform1
        self.transform2 = transform2

        self.imgs = []
        for img_name in self.img_names:
            img = Image.open(os.path.join(img_dir, img_name))

            if self.transform1 is not None:
                img = self.transform1(img)

            self.imgs.append(img)

    def __getitem__(self, index):
        img = self.imgs[index]

        if self.transform2 is not None:
            img = self.transform2(img)

        return img

    def __len__(self):
        return len(self.imgs)
