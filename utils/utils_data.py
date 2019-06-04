import os
import random

import numpy as np
import pandas as pd
from skimage import io
from torch.utils import data

ISIC_PATH = "/home/maparla/DeepLearning/Datasets/ISIC2019/"
ISIC_TRAIN_ROOT_PATH = ISIC_PATH + "ISIC_2019_Training_Input"

ROOT_PATH = {"": ISIC_TRAIN_ROOT_PATH, "train": ISIC_TRAIN_ROOT_PATH,
             "validation": ISIC_TRAIN_ROOT_PATH}

try:
    ISIC_TRAIN_DF_TRUTH = pd.read_csv(ISIC_PATH + "ISIC_2019_Training_GroundTruth_WithTargets.csv")
except:
    assert False, "You need to generate the ground truth with categorical targets! See EDA.ipynb!"
ISIC_TRAIN_DF_METADATA = pd.read_csv(ISIC_PATH + "ISIC_2019_Training_Metadata.csv")

DIAGNOSTIC_CATEGORIES = {"Melanoma": "MEL", "Melanocytic nevus": "NV", "Basal cell carcinoma": "BCC",
                         "Actinic keratosis": "AK", "Benign keratosis": "BKL", "Dermatofibroma": "DF",
                         "Vascular lesion": "VASC", "Squamous cell carcinoma": "SCC", "None of the others": "UNK"}
CATEGORIES_DIAGNOSTIC = {v: k for k, v in DIAGNOSTIC_CATEGORIES.items()}
CATEGORIES_CLASS = {"MEL": 0, "NV": 1, "BCC": 2, "AK": 3, "BKL": 4, "DF": 5, "VASC": 6, "SCC": 7, "UNK": 8}
CLASS_CATEGORIES = {v: k for k, v in CATEGORIES_CLASS.items()}


class ISIC2019_Dataset(data.Dataset):

    def __init__(self, data_partition="", transforms=None, albumentation=None, validation_size=0.15, seed=42):
        """
          - data_partition:
             -> Si esta vacio ("") devuelve todas las muestras de todo el TRAIN
             -> Si es "train" devuelve 1-validation_size muestras de todo el TRAIN
             -> Si es "validation" devuelve validation_size muestras de todo el TRAIN
        """
        self.root_path = ROOT_PATH[data_partition]
        self.imgs = []
        for dirpath, dirnames, files in os.walk(self.root_path):
            for f in files:
                if ".txt" not in f:
                    img = os.path.join(dirpath, f)
                    self.imgs.append(img)

        self.imgs = np.array(self.imgs)

        random.seed(seed)
        val_images = random.sample(range(len(self.imgs) + 1), int(validation_size * len(self.imgs)))

        if data_partition == "train":
            train_images = list(set(list(range(len(self.imgs)))) - set(val_images))
            self.imgs = self.imgs[train_images]
        elif data_partition == "validation":
            self.imgs = self.imgs[val_images]

        self.data_partition = data_partition
        self.albumentation = albumentation
        self.transform = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img_path = self.imgs[index]
        image = io.imread(img_path)

        img_name = img_path.split("/")[-1]
        img_name = img_name[:img_name.find(".jpg")]  # quitamos la extension del nombre

        target = ISIC_TRAIN_DF_TRUTH.loc[ISIC_TRAIN_DF_TRUTH.image == img_name].target.values[0]

        if self.transform:
            image = self.transform(image)

        if self.albumentation:
            try:
                augmented = self.albumentation_img(image=image)
                image = augmented['image']
            except:
                assert False, "Transform error in file: " + img_name

        return image, target
