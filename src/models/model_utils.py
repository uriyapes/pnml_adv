import torch
import torch.nn as nn
from abc import ABC
from utilities import TorchUtils


class ModelTemplate(nn.Module, ABC):
    def __init__(self):
        super(ModelTemplate, self).__init__()

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = True


def load_pretrained_model(model_base, model_params_path):
    device = TorchUtils.get_device()
    state_dict = torch.load(model_params_path, map_location=device)
    model_base.load_state_dict(state_dict)
    model_base = model_base.to(device)
    return model_base
