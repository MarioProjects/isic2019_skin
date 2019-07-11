import os
import random

import numpy as np
import pandas as pd
from skimage import io
from torch.utils import data
import torchy
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils.color_constancy as color_constancy

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

    elif type(norm) is int:
        feats = feats / norm

    elif norm == None or norm == "":
        pass
    else:
        assert False, "Data normalization not implemented: {}".format(norm)

    return feats


class ISIC2019_FromFolders(data.Dataset):

    def __init__(self, data_partition="", transforms=None, albumentation=None, normalize=255, seed=42, retinex=False, shade_of_gray=False):
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

        #random.seed(seed)
        #random.shuffle(self.imgs)

        self.data_partition = data_partition
        self.albumentation = albumentation
        self.transform = transforms
        self.normalize = normalize
        self.retinex = retinex
        self.shade_of_gray = shade_of_gray

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img_path = self.imgs[index]
        if self.retinex:
            img_path = img_path.replace("Train", "retinex_train")
        elif self.shade_of_gray:
            img_path = img_path.replace("Train", "shade_of_gray_train")

        image = io.imread(img_path)
        target = CATEGORIES_CLASS[img_path.split("/")[-2]]

        if self.albumentation:
            augmented = self.albumentation(image=image)
            image = augmented['image']

        if self.transform:
            image = self.transform(image)
        else: # If there are not Pytorch transformations, principally ToTensor(), we have to modify manually the data
            image = normalize_data(image, self.normalize) # ToTensor() transform Normalize 0-1
            image = image.transpose(2, 0, 1)  # Pytorch recibe en primer lugar los canales
        return image, target



def save_imgs(images, targets=None, display=False, save=True, custom_name="", num_test_samples=16, imgs_out_dir=""):
    fig=plt.figure(figsize = (7,7))
    gs1 = gridspec.GridSpec(4, 4)
    if targets is None:
        gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
    else:
        gs1.update(wspace=0.025, hspace=0.275) # set the spacing between axes.

    for i in range(num_test_samples):
        # i = i + 1 # grid spec indexes from 0
        ax1 = plt.subplot(gs1[i])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        if targets is not None:
            ax1.set_title(CLASS_CATEGORIES[targets[i].item()])

        if(images[i].shape[0]==1):
            img = images[i].reshape(images[i].shape[1], images[i].shape[2])
        elif(images[i].shape[0]==3):
            img = images[i].permute(1,2,0)
        else: assert False, "Check images dims!"
        plt.imshow(img)

    if display: plt.show()
    if save:
        if custom_name!="":
            fig.savefig(imgs_out_dir+custom_name+'.png', bbox_inches='tight')
        else: assert False, "Provide a name please!"
    plt.close()
