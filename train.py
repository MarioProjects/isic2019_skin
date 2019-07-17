# !/usr/bin/env python
# coding: utf-8

# ---- Library import ----

import pickle
from time import gmtime, strftime
import math

from sklearn.metrics import balanced_accuracy_score
import albumentations
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchy

from utils.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, random_search2048, \
    fa_reduced_imagenet, fa_reduced_cifar10

from utils.augmentations import *

class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


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

train_transforms = None

if args.pretrained_imagenet:
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


if args.data_augmentation:
    train_aug = albumentations.Compose([
        albumentations.PadIfNeeded(p=1, min_height=args.crop_size, min_width=args.crop_size),
        albumentations.Resize(args.img_size, args.img_size),
        albumentations.RandomCrop(p=1, height=args.crop_size, width=args.crop_size),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=0.22, contrast_limit=0.22),
        albumentations.HueSaturationValue(p=0.5, hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5),
        albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=45),
        albumentations.Cutout(p=0.5, num_holes=1, max_h_size=50, max_w_size=50)
    ])

    #train_transforms = transforms.Compose([
    #    transforms.ToTensor(),
    #])

    # fa_reduced_cifar10() - autoaug_paper_cifar10() - fa_reduced_imagenet()
    #train_transforms.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
    #train_transforms.transforms.insert(0, transforms.ToPILImage())


train_dataset = ISIC2019_FromFolders(data_partition="train", albumentation=train_aug, transforms=train_transforms,
                                     retinex=args.retinex, shade_of_gray=args.shade_of_gray, colornet=args.colornet)

if args.balanced_sampler:
    sampler_weights = get_sampler_weights()
    assert len(sampler_weights)==len(train_dataset), "Weights for data balancing not correspond to dataset"
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_weights, len(sampler_weights))
    train_loader = DataLoader(train_dataset, pin_memory=True, shuffle=False, sampler=sampler, batch_size=args.batch_size)
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, drop_last=True)

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
val_dataset = ISIC2019_FromFolders(data_partition="validation", albumentation=val_aug, transforms=val_transforms,
                                   retinex=args.retinex, shade_of_gray=args.shade_of_gray, colornet=args.colornet)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)
print("Data loaded!\n")

num_classes = len(np.unique(ISIC_TRAIN_DF_TRUTH.target))
print("{} Classes detected!".format(num_classes))
start_freezed = True if args.freezed_epochs > 0 else False
model = model_selector(args.model_name, num_classes, args.depth_coefficient, args.width_coefficient, freezed=start_freezed)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

progress_train_loss, progress_val_loss = [], []
progress_train_acc, progress_val_acc, progress_val_balanced_acc = [], [], []
best_loss, best_accuracy, best_balanced_accuracy, global_best_accuracy, global_best_balanced_accuracy = 10e10, -1, -1, -1, -1
alert_unfreeze = True

if args.weighted_loss:
    with open("class_weights.pkl", "rb") as fp:   # Unpickling
        weights = pickle.load(fp)
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)

if args.snapshot > 1:
    scheduler_step = args.epochs // args.snapshot
    print("Generating Snaps every {} epochs!".format(scheduler_step))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step)
    num_snapshot = 0
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 220, 270], gamma=0.1)

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

# ---- START TRAINING ----
print("\n---- Start Training ----")
for current_epoch in range(args.epochs):

    alert_unfreeze = check_unfreeze(alert_unfreeze, args.pretrained_imagenet, current_epoch, args.freezed_epochs, model, args.output_dir, args.model_name)

    if args.cutmix:
        train_loss, train_accuracy = train_step_cutmix(train_loader, model, criterion, optimizer)
        val_loss, val_accuracy, val_predicts, val_truths = torchy.utils.val_step(val_loader, model, criterion, data_predicts=True)
    if args.colornet:
        train_loss, train_accuracy = train_step_colornet(train_loader, model, criterion, optimizer)
        val_loss, val_accuracy, val_predicts, val_truths = val_step_colornet(val_loader, model, criterion, data_predicts=True)
    else:
        train_loss, train_accuracy = torchy.utils.train_step(train_loader, model, criterion, optimizer)
        val_loss, val_accuracy, val_predicts, val_truths = torchy.utils.val_step(val_loader, model, criterion,
                                                                                 data_predicts=True)

    val_balanced_accuracy = balanced_accuracy_score(val_truths, val_predicts)

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_accuracy_model = model.state_dict()
        if best_accuracy >= global_best_accuracy:
            global_best_accuracy = best_accuracy
            torch.save(model.state_dict(), args.output_dir + "model_" + args.model_name + "_GLOBAL_best_accuracy.pt")

    if val_balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = val_balanced_accuracy
        best_balanced_accuracy_model = model.state_dict()
        if best_balanced_accuracy >= global_best_balanced_accuracy:
            global_best_balanced_accuracy = best_balanced_accuracy
            torch.save(model.state_dict(), args.output_dir + "model_" + args.model_name + "_GLOBAL_best_balanced_accuracy.pt")

    # Imprimimos como va el entrenamiento
    current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("[{}] Epoch {}, LR: {:.6f}, Train Loss: {:.6f}, Val Loss: {:.6f}, Train Acc: {:.2f}, Val Acc: {:.4f}, Val Balanced Acc: {:.4f}".format(
        current_time, current_epoch + 1, torchy.utils.get_current_lr(optimizer), train_loss, val_loss, train_accuracy, val_accuracy, val_balanced_accuracy
    ))

    progress_train_loss.append(train_loss)
    progress_val_loss.append(val_loss)
    progress_train_acc.append(train_accuracy)
    progress_val_acc.append(val_accuracy)
    progress_val_balanced_acc.append(val_balanced_accuracy)

    torch.save(model.state_dict(), args.output_dir + "model_" + args.model_name + "_last.pt")

    progress = {"train_loss": progress_train_loss, "train_accuracy": progress_train_acc,
                "val_loss": progress_val_loss, "val_accuracy": progress_val_acc,
                "val_balanced_accuracy":progress_val_balanced_acc}
    with open(args.output_dir + 'progress.pickle', 'wb') as handle:
        pickle.dump(progress, handle, protocol=pickle.HIGHEST_PROTOCOL)

    scheduler.step()
    if args.snapshot > 1:
        if (current_epoch + 1) % scheduler_step == 0:
            print(" \n ------------------- SAVING SNAP: {} ------------------- ".format(num_snapshot))
            print("Val Acc: {:.4f}, Val Balanced Acc: {:.4f}\n".format(best_accuracy, best_balanced_accuracy))
            torch.save(best_accuracy_model,args.output_dir+"model_"+args.model_name+"_best_accuracy_snap"+str(num_snapshot)+".pt")
            torch.save(best_balanced_accuracy_model,args.output_dir+"model_"+args.model_name+"_best_balanced_accuracy_snap"+str(num_snapshot)+".pt")
            optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step)
            num_snapshot += 1
            best_accuracy, best_balanced_accuracy = -1, -1

print("\n------------------------")
print("Best Validation Accuracy {:.4f} at epoch {}".format(np.array(progress_val_acc).max(),
                                                           np.array(progress_val_acc).argmax() + 1))
print("Best Validation Balanced Accuracy {:.4f} at epoch {}".format(np.array(progress_val_balanced_acc).max(),
                                                           np.array(progress_val_balanced_acc).argmax() + 1))
print("------------------------\n")
