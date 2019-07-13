import os

import numpy as np
import os.path
import pathlib
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms, datasets

from resnet import load_pretrained_resnet32_cifar10_model
from resnet import resnet32
from mpl import Net, Net_800_400_100, load_pretrained_model
from utilities import TorchUtils


# Normalization for CIFAR10 dataset
mean_cifar10 = [0.485, 0.456, 0.406]
std_cifar10 = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean_cifar10, std=std_cifar10)


def create_adversarial_sign_dataset(data_folder='./data',
                                    output_folder=os.path.join('data', 'adversarial_sign'),
                                    model=load_pretrained_resnet32_cifar10_model(resnet32())):
    if os.path.exists(output_folder):
        return output_folder
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    testset = datasets.CIFAR10(root=data_folder,
                               train=False,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             normalize]))
    testloader = data.DataLoader(testset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0)

    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    # Accuracy counter
    criterion = nn.CrossEntropyLoss()
    # Loop over all examples in test set
    for iter_num, (image, target) in enumerate(testloader):
        # Send the image and label to the device
        image, target = TorchUtils.to_device(image), TorchUtils.to_device(target)

        # Set requires_grad attribute of tensor. Important for Attack
        image.requires_grad = True

        # Forward pass the image through the model
        output = model(image)
        loss = criterion(output, target)
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = image.grad.data

        # Collect the element-wise sign of the image gradient and save result
        sign_data_grad = data_grad.sign().squeeze()
        np.save(os.path.join(output_folder, str(iter_num)), sign_data_grad)
        if iter_num % 1000 == 0:
            print('Saved adversarial sign: ', iter_num)

    return output_folder


def create_adversarial_mnist_sign_dataset(data_dir, load_sign_dataset, adversarial_sign_dataset_path, create_sign_dataset_model_path, normalize_mnist):
    if load_sign_dataset:
        assert(os.path.exists(adversarial_sign_dataset_path))
        print("Load:" + adversarial_sign_dataset_path)
        return

    # model = Net()
    model = Net_800_400_100()
    model = load_pretrained_model(model, create_sign_dataset_model_path)

    print("create_adversarial_mnist_sign_dataset from model:" + create_sign_dataset_model_path)
    pathlib.Path(adversarial_sign_dataset_path).mkdir(parents=True, exist_ok=True)
    testset = datasets.MNIST(root=data_dir,
                             train=False,
                             download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           normalize_mnist]))
    testloader = data.DataLoader(testset,
                                 batch_size=128,
                                 shuffle=False,
                                 num_workers=0)

    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    # Accuracy counter
    criterion = nn.CrossEntropyLoss()
    # Loop over all examples in test set
    for iter_num, (image, target) in enumerate(testloader):
        # Send the image and label to the device
        image, target = TorchUtils.to_device(image), TorchUtils.to_device(target)

        # Set requires_grad attribute of tensor. Important for Attack
        image.requires_grad = True

        # Forward pass the image through the model
        output = model(image)
        loss = criterion(output, target) * len(image)
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = image.grad.data

        # Collect the element-wise sign of the image gradient and save result
        sign_data_grad = data_grad.sign().squeeze()
        np.save(os.path.join(adversarial_sign_dataset_path, str(iter_num)), sign_data_grad)
        if iter_num % 500 == 0:
            print('Saved adversarial sign: ', iter_num)
