
#!/usr/bin/env python
# coding: utf-8

# ---- Library import ----

from time import gmtime, strftime

import torch
from torch.utils.data import DataLoader


import albumentations

# ---- My utils ----
from utils.utils_data import *
from utils.train_arguments import *

EFFICIENTNET_SEARCH_SPACE = [
    # Depth | Width | Resolution
    # D*W^2*R^2 aprox 2
    [1, 1, 1],
    [1.2, 1.1, 1.15],
    [1.1, 1.2, 1.15],
    [1.3, 1.15, 1.1],
    [1, 1.1, 1.3],
    [1, 1.3, 1.1],
    [1.1, 1.05, 1.3],
    [1.35, 1.1, 1.1],
    [1.1, 1.3, 1.05],
    [1.1, 1.05, 1.3]
]


train_aug = albumentations.Compose([
                albumentations.PadIfNeeded(p=1, min_height=args.crop_size, min_width=args.crop_size),
                albumentations.RandomCrop(p=1, height=args.crop_size, width=args.crop_size),
                albumentations.Resize(args.img_size, args.img_size)
            ])

val_aug = albumentations.Compose([
                albumentations.PadIfNeeded(p=1, min_height=args.crop_size, min_width=args.crop_size),
                albumentations.Resize(args.img_size, args.img_size),
                albumentations.CenterCrop(p=1, height=args.crop_size, width=args.crop_size)
          ])

if args.data_augmentation:
    print("Data Augmentation to be implemented...")


train_dataset = ISIC2019_Dataset(data_partition="train", transforms=train_aug)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

val_dataset = ISIC2019_Dataset(data_partition="validation", transforms=val_aug)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)


model = model_selector(args.model_name)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
