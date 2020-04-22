import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC
from torchvision import models
from utilities import TorchUtils
from torchvision.transforms import functional as trans_functional


class ModelTemplate(nn.Module, ABC):
    def __init__(self):
        super(ModelTemplate, self).__init__()
        self.pnml_model = False
        self.regularization = TorchUtils.to_device(torch.zeros([1]))

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = True

    def calc_log_prob(self, x):
        return F.log_softmax(self.__call__(x), dim=1)

    def get_genie_prob(self):
        return None

    def eval_batch(self, data, labels, enable_grad: bool = True):
        """
        :param data: the data to evaluate
        :param labels: the labels of the data
        :param enable_grad: Should grad be enabled for later differentiation
        :param model_output_type: "logits" if model output logits or "prob"
        :param loss_func: The loss function used to calculate the loss
        :return: batch loss, probability and label prediction.
        """
        loss_func = torch.nn.NLLLoss(reduction='none')
        self.eval()
        with torch.set_grad_enabled(enable_grad):
            data, labels = TorchUtils.to_device(data), TorchUtils.to_device(labels)
            output = self.__call__(data)
            if not self.pnml_model:  # Output is logits
                prob = torch.softmax(output, 1)
                loss = loss_func(torch.log_softmax(output, 1), labels)  # log soft-max is more numerically stable
            else:  # Output is probability
                prob = output
                loss = loss_func(torch.log(output), labels)
            # _, predicted_label = torch.max(output.data, 1)
        return loss, prob, self.get_genie_prob()


def load_pretrained_model(model_base, model_params_path):
    device = TorchUtils.get_device()
    state_dict = torch.load(model_params_path, map_location=device)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    model_base.load_state_dict(state_dict)
    model_base = model_base.to(device)
    return model_base


def per_image_standardization_tf(x: torch.Tensor):
    """Linearly scales `image` to have zero mean and unit variance.
    This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
    of all values in image, and `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`. Where NumElements
    include the pixels in all channels combined.
    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.
    Args:
      x: An N-D Tensor where the first dimension is the batch dimension and the others are [Channels, Height, Width]
    Returns:
      The standardized image with same shape as `image`.
    """
    dim = x.dim()
    assert(dim == 4)  # Untested with different number of channels but could work
    assert(x.size()[1] == 3)  # make sure first dim is channels RGB
    pix_num = torch.prod(torch.tensor(x.shape)[-dim+1:], dtype=torch.float).to(x.device)
    min_stddev = 1 / torch.sqrt(pix_num)
    mean = x
    # Calculates mean over each image in the batch. In pytorch > 0.4.1 could be done without for loop
    for dim_to_reduce in range(1, dim):
        mean = torch.mean(mean, dim_to_reduce, keepdim=True)
    std = torch.sqrt((1/pix_num) * torch.sum(x**2, dim=(1, 2, 3), keepdim=True) - (mean**2))
    adjusted_stddev = torch.max(std, min_stddev)
    return (x - mean) / adjusted_stddev


class AddNoiseTransform(object):
    """Adds gaussian noise to the image
    Args:
      x: An N-D Tensor where the first dimension is the batch dimension and the others are [Channels, Height, Width]
    Returns:
      The noisy image with same shape as `image`.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        # Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval
        noisy_x = x + 3 * torch.rand_like(x)
        # noisy_x = x + np.random.normal(0,1,x.shape)
        return noisy_x


class NormalizeCls(nn.Module):
    def __init__(self, mean: list, std: list):
        """
        :param mean: a list representing the mean values in each channel
        :param std: a list representing the std values in each channel
        """
        super(NormalizeCls, self).__init__()
        assert(type(mean) == list and type(std) == list and len(mean) == len(std))
        channels = len(mean)
        self.mean = TorchUtils.to_device(torch.tensor(mean, dtype=torch.float).reshape([channels, 1, 1]))
        self.std = TorchUtils.to_device(torch.tensor(std, dtype=torch.float).reshape([channels, 1, 1]))

    def forward(self, tensor):
        """
        Args:
            tensor: Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return (tensor - self.mean) / self.std


class ImagenetModel(ModelTemplate):
    def __init__(self, existing_model, num_classes: int = 1000):
        super(ImagenetModel, self).__init__()
        self.existing_model = existing_model
        self.normalization = NormalizeCls(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_classes = num_classes

    def forward(self, x):
        x_norm = self.normalization(x)
        out = self.existing_model(x_norm)[:, :self.num_classes]
        return out


def load_pretrained_imagenet_model(model_name:str, pretrained: bool = True):
    """
    :param model_name: Could be one of the following:
        'alexnet',
     'densenet',
         'densenet121',
         'densenet161',
         'densenet169',
         'densenet201',
     'inception',
         'inception_v3',
     'resnet',
         'resnet101',
         'resnet152',
         'resnet18',
         'resnet34',
         'resnet50',
     'squeezenet',
         'squeezenet1_0',
         'squeezenet1_1',
     'vgg',
         'vgg11',
         'vgg11_bn',
         'vgg13',
         'vgg13_bn',
         'vgg16',
         'vgg16_bn',
         'vgg19',
         'vgg19_bn'
    :param: pretrained: Whether to return a pretrained model or not.
    :return: the trained model
    """
    model = getattr(models, model_name)(pretrained=pretrained)  # equivalent to models.resnet101(pretrained=True) for example
    # model_with_norm = ImagenetModel(model)
    return TorchUtils.to_device(model)