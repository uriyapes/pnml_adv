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
from mpl import Net


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accuracy counter
    criterion = nn.CrossEntropyLoss()
    # Loop over all examples in test set
    for iter_num, (image, target) in enumerate(testloader):
        # Send the image and label to the device
        image, target = image.to(device), target.to(device)

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


def create_adversarial_mnist_sign_dataset(data_folder='./data',
                                    output_folder=os.path.join('data', 'adversarial_sign_mnist')):
    model = Net()
    model.load_state_dict(torch.load("./models/pnml_mnist_model_0.163129.pt"))
    normalize_mnist = transforms.Normalize(mean=[0.1307], std=[0.3081])
    if os.path.exists(output_folder):
        return output_folder
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    testset = datasets.MNIST(root=data_folder,
                             train=False,
                             download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           normalize_mnist]))
    testloader = data.DataLoader(testset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0)

    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accuracy counter
    criterion = nn.CrossEntropyLoss()
    # Loop over all examples in test set
    for iter_num, (image, target) in enumerate(testloader):
        # Send the image and label to the device
        image, target = image.to(device), target.to(device)

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