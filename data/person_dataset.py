import cv2
import torch
import torch.utils.data as Data
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm
from albumentations import *
from utils.augmentation_utils import letterbox_image
from utils.augmentation_utils import crop_image

from albumentations.pytorch import ToTensor
from skimage.io import imshow, imread, imsave

import warnings
warnings.filterwarnings("ignore")

class PersonDataset(Data.Dataset):
    def __init__(self, image_paths, labels = None, train = True, test = False):
        self.paths = image_paths
        self.test = test
        if self.test == False:
            self.labels = labels
        self.train = train
        self.train_transform = Compose([# CenterCrop(height=1365, width=1365, p=1),
                                        Resize(height=300, width=300, interpolation=1, p=1),
                                        HorizontalFlip(p=0.5),
                                        VerticalFlip(p=0.5),
                                        ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                                        OneOf([IAAEmboss(p=1), IAASharpen(p=1), Blur(p=1)], p=0.5),
                                        IAAPiecewiseAffine(p=0.5)
        ])
        self.test_transform = Compose([# CenterCrop(height=1365, width=1365, p=1),
                                       Resize(height=300, width=300, interpolation=1, p=1),
                                       HorizontalFlip(p=0.5),
                                       VerticalFlip(p=0.5),
                                       ShiftScaleRotate(rotate_limit=25.0, p=0.7)
        ])
        self.default_transform = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),
                                          ToTensor()])
    def __len__(self):
        return self.paths.shape[0]
    
    def __getitem__(self, i):
        image = imread(self.paths[i])
        # image = letterbox_image(image, (300, 300))
        # image = crop_image(image)
    
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2).repeat(3, 2)
        elif image.shape[2] > 3:  # test/1078.jpg
            image = image[:, :, :3]

        if self.test==False:
            label = torch.tensor(np.argmax(self.labels.loc[i,:].values))
        if self.train:
            image = self.train_transform(image=image)['image']
            image = self.default_transform(image=image)['image']
        elif self.test:
            image = self.test_transform(image=image)['image']
            image = self.default_transform(image=image)['image']
        else:
            image = self.default_transform(image=image)['image']

        if self.test==False:
            return image, label
        return image
