import torch
import torchy

def model_selector(model_name, depth_coefficient=1.0, width_coefficient=1.0):
    if model_name == "efficientnet":
        return torchy.models.EfficientNet_Constants(depth_coefficient, width_coefficient).cuda()
    else:
        assert False, "Uknown model selected!"

def get_optimizer(optmizer_type, model, lr=0.1):
    if optmizer_type=="sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optmizer_type=="adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0002)
    elif optmizer_type=="rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)

    assert False, 'No optimizers with that name! ['+optmizer_type+']'