import torch
import torch.nn as nn
import torchy
from efficientnet_pytorch import EfficientNet


def model_selector(model_name, num_classes, depth_coefficient=1.0, width_coefficient=1.0, freezed=True):
    if model_name == "efficientnet":
        return torchy.models.EfficientNet_Constants(depth_coefficient, width_coefficient, num_classes=num_classes).cuda()
    # https://github.com/lukemelas/EfficientNet-PyTorch#loading-pretrained-models
    elif model_name == "efficientnet_pretrained_b5":
        model = EfficientNet.from_pretrained('efficientnet-b5')
        if freezed:
            for param in model.parameters():
                param.requires_grad = False
        model._fc = nn.Linear(2048, num_classes)
        return model
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
