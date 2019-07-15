import torch
import torch.nn as nn
import torchy
from efficientnet_pytorch import EfficientNet
from pytorchcv.model_provider import get_model as ptcv_get_model
import models
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_selector(model_name, num_classes, depth_coefficient=1.0, width_coefficient=1.0, freezed=True):
    if model_name == "efficientnet":
        return torchy.models.EfficientNet_Constants(depth_coefficient, width_coefficient, num_classes=num_classes).cuda()
    # https://github.com/lukemelas/EfficientNet-PyTorch#loading-pretrained-models
    elif model_name == "efficientnet_pretrained_b4":
        model = EfficientNet.from_pretrained('efficientnet-b4')
        if freezed:
            for param in model.parameters():
                param.requires_grad = False
        model._fc = nn.Linear(1792, num_classes)
        return model.cuda()
    elif model_name == "efficientnet_pretrained_b5":
        model = EfficientNet.from_pretrained('efficientnet-b5')
        if freezed:
            for param in model.parameters():
                param.requires_grad = False
        model._fc = nn.Linear(2048, num_classes)
        return model.cuda()
    elif model_name == "resnet34":
        return torchy.models.ResNet(torchy.models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes).cuda()
    elif model_name == "resnet50":
        return torchy.models.ResNet(torchy.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes).cuda()
    elif model_name == "se_resnext101_32x4d_pretrained":
        model = ptcv_get_model("seresnext101_32x4d", pretrained=True)
        if freezed:
            for param in model.parameters():
                param.requires_grad = False
        dim_feats = model.last_linear.in_features  # =2048
        model.last_linear = nn.Linear(dim_feats, num_classes)
        return model.cuda()
    elif model_name == "se_resnext101_32x4d":
        model = ptcv_get_model("seresnext101_32x4d", pretrained=False, num_classes=num_classes)
        return model.cuda()
    elif model_name == "seresnext50_32x4d":
        model = ptcv_get_model("seresnext50_32x4d", pretrained=False, num_classes=num_classes)
        return model.cuda()
    elif model_name == "resnetd50b":
        model = ptcv_get_model("resnetd50b", pretrained=False, num_classes=num_classes)
        return model.cuda()
    elif model_name == "resnetd101b":
        model = ptcv_get_model("resnetd101b", pretrained=False, num_classes=num_classes)
        return model.cuda()
    elif model_name == "resnetd152b":
        model = ptcv_get_model("resnetd152b", pretrained=False, num_classes=num_classes)
        return model.cuda()
    elif "color-densenet-40" in model_name:
        growth_rate = int(model_name.split("-")[-1])
        return models.colornet.ColorNet_40_x(growth_rate=growth_rate, num_classes=num_classes).cuda()
    else:
        assert False, "Uknown model selected!"


def get_optimizer(optmizer_type, model, lr=0.1):
    if optmizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optmizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0002)
    elif optmizer_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)

    assert False, 'No optimizers with that name! [' + optmizer_type + ']'


def check_unfreeze(alert_unfreeze, pretrained_imagenet, current_epoch, freezed_epochs, model, output_dir, model_name):
    if alert_unfreeze and pretrained_imagenet and current_epoch >= freezed_epochs:
        print("\n------- UNFREEZE MODEL -------\n")
        for param in model.parameters():
            param.requires_grad = True

        torch.save(model.state_dict(), output_dir + "model_" + model_name + "_FreezedPhase.pt")
        return False # Model unfreezed
    return True # Model Not unfreezed



def train_step_colornet(train_loader, model, criterion, optimizer):
    train_loss, train_correct = [], 0
    model.train()
    ###for rgb, lab, hsv, yuv, ycbcr, hed, yiq, target in train_loader:
    for rgb, lab, hsv, target in train_loader:

        ###rgb, lab, hsv, yuv, ycbcr, hed, yiq, target = rgb.to(DEVICE), lab.to(DEVICE), hsv.to(DEVICE), yuv.to(DEVICE), ycbcr.to(DEVICE), hed.to(DEVICE), yiq.to(DEVICE), target.to(DEVICE)
        ###rgb, lab, hsv, yuv, ycbcr, hed, yiq = rgb.type(torch.float), lab.type(torch.float), hsv.type(torch.float), yuv.type(torch.float), ycbcr.type(torch.float), hed.type(torch.float), yiq.type(torch.float)
        ###y_pred = model(rgb, lab, hsv, yuv, ycbcr, hed, yiq)

        rgb, lab, hsv, target = rgb.to(DEVICE), lab.to(DEVICE), hsv.to(DEVICE), target.to(DEVICE)
        rgb, lab, hsv = rgb.type(torch.float), lab.type(torch.float), hsv.type(torch.float)
        y_pred = model(rgb, lab, hsv)

        loss = criterion(y_pred.float(), target.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = y_pred.max(1)  # get the index of the max log-probability
        train_correct += pred.eq(target).sum().item()
        train_loss.append(loss.item())

    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    return np.mean(train_loss), train_accuracy


def val_step_colornet(val_loader, model, criterion, data_predicts = False):
    val_loss, val_correct = [], 0
    predicts, truths = [], []
    init = -1
    model.eval()
    with torch.no_grad():
        ###for rgb, lab, hsv, yuv, ycbcr, hed, yiq, target in val_loader:
        for rgb, lab, hsv, target in val_loader:

            ###rgb, lab, hsv, yuv, ycbcr, hed, yiq, target = rgb.to(DEVICE), lab.to(DEVICE), hsv.to(DEVICE), yuv.to(DEVICE), ycbcr.to(DEVICE), hed.to(DEVICE), yiq.to(DEVICE), target.to(DEVICE)
            ###rgb, lab, hsv, yuv, ycbcr, hed, yiq = rgb.type(torch.float), lab.type(torch.float), hsv.type(torch.float), yuv.type(torch.float), ycbcr.type(torch.float), hed.type(torch.float), yiq.type(torch.float)
            ###y_pred = model(rgb, lab, hsv, yuv, ycbcr, hed, yiq)

            rgb, lab, hsv, target = rgb.to(DEVICE), lab.to(DEVICE), hsv.to(DEVICE), target.to(DEVICE)
            rgb, lab, hsv = rgb.type(torch.float), lab.type(torch.float), hsv.type(torch.float)
            y_pred = model(rgb, lab, hsv)

            loss = criterion(y_pred.float(), target.long())
            val_loss.append(loss.item())
            _, pred = y_pred.max(1)  # get the index of the max log-probability
            val_correct += pred.eq(target).sum().item()
            if data_predicts:
                predicts.append(pred.detach().cpu().numpy())
                truths.append(target.detach().cpu().numpy())
 

    val_accuracy = 100. * val_correct / len(val_loader.dataset)
    if not data_predicts: return np.mean(val_loss), val_accuracy

    predicts = np.concatenate(predicts)
    truths = np.concatenate(truths)
    return np.mean(val_loss), val_accuracy, predicts, truths
