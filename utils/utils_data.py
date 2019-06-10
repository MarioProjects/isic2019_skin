import os
import random

import numpy as np
import pandas as pd
from skimage import io
from torch.utils import data
import torchy
import pickle

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

VALIDATION_FILE = ISIC_PATH + "valid.txt"
VALIDATION_IMGS = open(VALIDATION_FILE).read().split('\n')
for indx, img in enumerate(VALIDATION_IMGS):
    # Corregimos el path para que sea absoluto
    if '.jpg' in img:
        VALIDATION_IMGS[indx] = ISIC_PATH + "Train/" + "/".join(img.split("/")[1:])
VALIDATION_IMGS = list(filter(None, VALIDATION_IMGS)) # Sanity check no empty lines/items in list

TRAIN_FILE = ISIC_PATH + "train.txt"
TRAIN_IMGS = open(TRAIN_FILE).read().split('\n')
for indx, img in enumerate(TRAIN_IMGS):
    # Corregimos el path para que sea absoluto
    if '.jpg' in img:
        TRAIN_IMGS[indx] = ISIC_PATH + "Train/" + "/".join(img.split("/")[1:])
TRAIN_IMGS = list(filter(None, TRAIN_IMGS)) # Sanity check no empty lines/items in list


def get_sampler_weights():
    if not os.path.exists("weights_sampler.pickle"):
        TRAIN_REAL_INDEXES = []
        for train_img in TRAIN_IMGS:
            img = train_img[train_img.find("ISIC_"):train_img.find(".jpg")]
            real_index = ISIC_TRAIN_DF_TRUTH.loc[ISIC_TRAIN_DF_TRUTH['image'] == img].index.values.astype(int)[0]
            TRAIN_REAL_INDEXES.append(real_index)
        torchy.utils.create_sampler_weights(ISIC_TRAIN_DF_TRUTH.loc[TRAIN_REAL_INDEXES], "target", "weights_sampler.pickle")
    with open('weights_sampler.pickle', 'rb') as fp:
        sampler_weights = pickle.load(fp)
    return sampler_weights


def normalize_data(feats, norm):
    if norm == '0_1range':
        max_val = feats.max()
        min_val = feats.min()
        feats = feats - min_val
        feats = feats / (max_val - min_val)


    elif norm == '-1_1range' or norm == 'np_range':
        max_val = feats.max()
        min_val = feats.min()
        feats = feats * 2
        feats = feats - (max_val - min_val)
        feats = feats / (max_val - min_val)

    elif norm == '255':
        feats = feats / 255

    elif norm == None or norm == "":
        pass
    else:
        assert False, "Data normalization not implemented: {}".format(norm)

    return feats

class ISIC2019_FromFolders(data.Dataset):

    def __init__(self, data_partition="", transforms=None, albumentation=None, normalize="255"):
        """
          - data_partition:
             -> Si esta vacio ("") devuelve todas las muestras de todo el TRAIN
             -> Si es "train" devuelve 85% muestras de todo el TRAIN
             -> Si es "validation" devuelve 15% muestras de todo el TRAIN
        """
        self.root_path = ROOT_PATH[data_partition]

        if data_partition == "train":
            self.imgs = TRAIN_IMGS
        elif data_partition == "validation":
            self.imgs = VALIDATION_IMGS

        self.data_partition = data_partition
        self.albumentation = albumentation
        self.transform = transforms
        self.normalize = normalize

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img_path = self.imgs[index]
        try:
            image = io.imread(img_path)
        except:
            assert False, "Error in file: {}".format(img_path)
        target = CATEGORIES_CLASS[img_path.split("/")[-2]]

        if self.albumentation:
            try:
                augmented = self.albumentation(image=image)
                image = augmented['image']
            except:
                assert False, "Transform error in file: " + img_path

        if self.transform:
            image = self.transform(image)

        #print(image.max())
        #image = normalize_data(image, self.normalize) # ToTensor() transform Normalize 0-1
        #image = image.transpose(2, 0, 1)  # Pytorch recibe en primer lugar los canales
        return image, target


class ISIC2019_Dataset(data.Dataset):
    ### -----------------------------------------------
    ### DEPRECATED! -> USE INSTEAD ISIC2019_FromFolders
    ### -----------------------------------------------
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
        val_images = random.sample(range(len(self.imgs)), int(validation_size * len(self.imgs)))

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
                augmented = self.albumentation(image=image)
                image = augmented['image']
            except:
                assert False, "Transform error in file: " + img_name

        image = image.transpose(2, 0, 1)  # Pytorch recibe en primer lugar los canales
        return image, target
