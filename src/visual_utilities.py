import matplotlib.pyplot as plt
from dataset_utilities import *
import numpy as np


print(os.getcwd()) #print working dir
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

def plt_img(image_batch, index=0):
    """
    plt_img plot image_batch[index]
    :param image_batch: a 4 dimension PIL.Image /np.ndarry / torch.Tensor which represents a batch of images
    :param index: the index of the image to plot
    :return: 
    """
    plt.figure()
    # convert to np image
    if type(image_batch) == np.ndarray:
        img = image_batch[index]
    elif type(image_batch) == torch.Tensor:
        # remove redundant dimension for grayscale and transform to numpy
        img = (image_batch[index].cpu().detach().squeeze().numpy())
    if img.shape[0] == 3:
        img = np.moveaxis(img, (0,1,2), (2,0,1))  # Make img shape HxWxC
        print(img.shape)
    plt.imshow(img, cmap='gray')  #  cmap ignored if img is 3-D
    plt.show()

