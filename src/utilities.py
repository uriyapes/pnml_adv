from abc import ABC, abstractmethod
import matplotlib
import matplotlib.pyplot as plt
from typing import Callable
import numpy as np
import torch
import os
import random


# print(os.getcwd()) #print working dir
def plt_adv_mnist_img(testloader = None):
    # trainloader, testloader, classes = create_adversarial_mnist_dataloaders(data_dir='./data', adversarial_dir='./data/mnist_adversarial_sign_batch', epsilon=0.25)

    num_of_img_to_plt = 10
    for batch_idx, (inputs, labels) in enumerate(testloader):
        i = 0
        for i in range(min(inputs.shape[0], num_of_img_to_plt)):
            plt.figure()
            img = (inputs.numpy())[i,0]
            plt.imshow(img, cmap='gray')
            plt.show()
        break


def plt_img(image_batch, index_list=[0], is_save_fig=False):
    """
    plt_img plot image_batch[index]
    :param image_batch: a 4 dimension PIL.Image /np.ndarry / torch.Tensor which represents a batch of images
    :param index: the index of the image to plot
    :return: 
    """
    plt.figure()
    ncols = 1
    nrows = max(1, int(len(index_list)/ncols))
    fig = plt.figure()
    # gs = matplotlib.gridspec.GridSpec(nrows, ncols, figure=fig)
    # f, axarr = plt.subplots(2, 2)
    for i, img_index in enumerate(index_list):
        subplot = fig.add_subplot(nrows, ncols, i+1)
        # subplot = fig.add_subplot(gs[i % nrows, int(i/nrows)])
        # convert to np image
        if type(image_batch) == np.ndarray:
            img = image_batch[img_index]
        elif type(image_batch) == torch.Tensor:
            # remove redundant dimension for grayscale and transform to numpy
            img = (image_batch[img_index].cpu().detach().squeeze().numpy())
        if img.shape[0] == 3:
            img = np.moveaxis(img, (0,1,2), (2,0,1))  # Make img shape HxWxC
            # print(img.shape)
        subplot.imshow(img, cmap='gray')  #  cmap ignored if img is 3-D
        subplot.axis('off')
    plt.savefig('./adv_output_images.jpg', bbox_inches=plt.tight_layout()) if is_save_fig else None
    plt.show()

class TorchUtils(ABC):
    __device__ = None
    __seed__ = None

    @classmethod
    @abstractmethod
    def to_device(cls, tensor: torch.tensor):
        if cls.__device__ is None:
            cls._auto_select_device()
        return tensor.to(cls.get_device())

    @classmethod
    @abstractmethod
    def set_device(cls, device):
        """
        Set the device that will do the calculations. This method should be called once in the start of your code.
        :param device: 'cuda' for GPU, 'cpu' or None
        :return:
        """
        if cls.__device__ is None:
            cls.__device__ = device
        else:
            raise Exception("device was already set")

    @classmethod
    @abstractmethod
    def get_device(cls):
        if cls.__device__ is None:
            cls._auto_select_device()
        return cls.__device__

    @classmethod
    @abstractmethod
    def _auto_select_device(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Setting device for pytorch: " + device)
        cls.set_device(device)

    @classmethod
    @abstractmethod
    def set_rnd_seed(cls, seed: int):
        # Important note: When using multiple workers dataloader(num_workers > 1) you need to create DataLoader with
        # worker_init_fn=set_rnd_seed(<your_seed>)
        assert(type(seed) == int)
        cls.__seed__ = seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
