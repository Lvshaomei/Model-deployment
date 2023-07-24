import cv2
from PIL import Image
from torch.utils.data import Dataset
import random
import PIL.ImageOps
import numpy as np
import torch


class my_datasets_process(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform1 = transform
        self.transform2 = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if len(self.transform1)>0:
            img0 = np.array(img0)
            # cv2.imshow('img0',img0)
            # cv2.waitKey(0)
            # tt = self.transform1[0]
            img0 = self.transform1[0](image = img0)['image']
            img0 = Image.fromarray(img0)
            img0 = self.transform1[1](img0)

            # cv2.imshow('img0',cv2.cvtColor(np.array(img0),cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            img1 = np.array(img1)
            img1 = self.transform2[0](image = img1)['image']
            img1 = Image.fromarray(img1)
            img1 = self.transform2[1](img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] == img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
