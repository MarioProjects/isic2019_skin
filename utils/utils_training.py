import torch
import torch.nn as nn
import torchy
from efficientnet_pytorch import EfficientNet


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
    else:
        assert False, "Uknown model selected!"


def get_optimizer(optmizer_type, model, lr=0.1):
    if optmizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optmizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0002)
    elif optmizer_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)

    assert False, 'No optimizers with that name! [' + optmizer_type + ']'


def check_unfreeze(alert_unfreeze, pretrained_imagenet, current_epoch, freezed_epochs, model):
    if alert_unfreeze and pretrained_imagenet and current_epoch >= freezed_epochs:
        print("\n------- UNFREEZE MODEL -------\n")
        for param in model.parameters():
            param.requires_grad = True
        return False # Model unfreezed
    return True # Model Not unfreezed