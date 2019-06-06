# !/usr/bin/env python
# coding: utf-8

# ---- Library import ----

import pickle
from time import gmtime, strftime

import albumentations
import math
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- My utils ----
from utils.train_arguments import *
from utils.utils_data import *
from utils.utils_training import *

# Primero necesitamos reescalar (si usamos los coeficientes de Efficientnet) la resolucion de las imagenes a usar
args.crop_size = math.ceil(args.crop_size * args.resolution_coefficient)
args.img_size = math.ceil(args.img_size * args.resolution_coefficient)

train_aug = albumentations.Compose([
    albumentations.PadIfNeeded(p=1, min_height=args.crop_size, min_width=args.crop_size),
    albumentations.Resize(args.img_size, args.img_size),
    albumentations.RandomCrop(p=1, height=args.crop_size, width=args.crop_size)
])

val_aug = albumentations.Compose([
    albumentations.PadIfNeeded(p=1, min_height=args.crop_size, min_width=args.crop_size),
    albumentations.Resize(args.img_size, args.img_size),
    albumentations.CenterCrop(p=1, height=args.crop_size, width=args.crop_size)
])

if args.data_augmentation:
    print("Data Augmentation to be implemented...")

train_dataset = ISIC2019_Dataset(data_partition="train", albumentation=train_aug)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

val_dataset = ISIC2019_Dataset(data_partition="validation", albumentation=val_aug)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)
print("Data loaded!\n")

num_classes = len(np.unique(ISIC_TRAIN_DF_TRUTH.target))
print("{} Classes detected!".format(num_classes))
model = model_selector(args.model_name, num_classes, args.depth_coefficient, args.width_coefficient)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

progress_train_loss, progress_val_loss, progress_train_acc, progress_val_acc = [], [], [], []
best_loss, best_accuracy = 10e10, -1

criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 220, 270], gamma=0.15)

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

# ---- START TRAINING ----
print("\n---- Start Training ----")
for current_epoch in range(args.epochs):

    train_loss, train_accuracy = torchy.utils.train_step(train_loader, model, criterion, optimizer)

    val_loss, val_accuracy = torchy.utils.val_step(val_loader, model, criterion)

    if val_loss > best_loss:
        torch.save(model.state_dict(), args.output_dir + "model_" + args.model_name + "_best_loss.pt")
        best_loss = val_loss

    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), args.output_dir + "model_" + args.model_name + "_best_accuracy.pt")
        best_accuracy = val_accuracy

    # Imprimimos como va el entrenamiento
    current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("[{}] Epoch {}, LR: {:.6f}, Train Loss: {:.6f}, Val Loss: {:.6f}, Train Acc: {:.2f}, Val Acc: {:.2f}".format(
        current_time, current_epoch + 1, torchy.utils.get_current_lr(optimizer), train_loss, val_loss, train_accuracy, val_accuracy
    ))

    progress_train_loss.append(train_loss)
    progress_val_loss.append(val_loss)
    progress_train_acc.append(train_accuracy)
    progress_val_acc.append(val_accuracy)

    torch.save(model.state_dict(), args.output_dir + "model_" + args.model_name + "_last.pt")

    progress = {"train_loss": progress_train_loss, "train_accuracy": progress_train_acc,
                "val_loss": progress_val_loss, "val_accuracy": progress_val_acc}
    with open(args.output_dir + 'progress.pickle', 'wb') as handle:
        pickle.dump(progress, handle, protocol=pickle.HIGHEST_PROTOCOL)

    scheduler.step()

print("\n------------------------")
print("Best Validation Accuracy {:.4f} at epoch {}".format(np.array(progress_val_acc).max(),
                                                           np.array(progress_val_acc).argmax() + 1))
print("------------------------\n")
