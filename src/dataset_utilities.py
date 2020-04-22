import copy
import os
import time
import numpy as np
import os.path
import torch
from torch.utils import data
from torchvision import transforms, datasets
from PIL import Image
from utilities import TorchUtils

# Normalization for CIFAR10 dataset
mean_cifar10 = (0.4914, 0.4822, 0.4465),
std_cifar10 = (0.2023, 0.1994, 0.2010)
normalize_cifar = transforms.Normalize(mean=mean_cifar10, std=std_cifar10)
mnist_std = 0.3081
mean_mnist = 0.1307
normalize_mnist = transforms.Normalize(mean=[mean_mnist], std=[mnist_std])
mnist_min_val = (0 - mean_mnist) / mnist_std
mnist_max_val = (1 - mean_mnist) / mnist_std
assert(mnist_max_val > mnist_min_val)
assert(np.isclose(1/mnist_std, mnist_max_val - mnist_min_val))
cifar_min_val = 0
cifar_max_val = 1
assert(cifar_max_val > cifar_min_val)
imagenet_min_val = 0
imagenet_max_val = 1

shuffle_train_set = True


def get_dataset_min_max_val(dataset_name: str, dtype=None):
    def _calc_normalized_val(val, mean, std, np_type):
        assert(np_type == np.float32 or np_type == np.float64 or np_type is None)
        if np_type is not None:
            val = np.array([val], dtype=np_type)  # Calculate the normalized value with a certain dtype
        return (val - mean) / std

    if dataset_name == 'cifar_adversarial':
        return cifar_min_val, cifar_max_val
    elif dataset_name == 'mnist_adversarial':
        return _calc_normalized_val(0, mean_mnist, mnist_std, dtype), _calc_normalized_val(1, mean_mnist, mnist_std, dtype)
    elif dataset_name == 'imagenet':
        return imagenet_min_val, imagenet_max_val
    else:
        raise NameError("No experiment name:" + dataset_name)


def insert_sample_to_dataset(trainloader, sample_to_insert_data, sample_to_insert_label):
    """
    Inserting test sample into the trainset
    :param trainloader: contains the trainset
    :param sample_to_insert_data: the data which we want to insert
    :param sample_to_insert_label: the data label which we want to insert
    :return: dataloader which contains the trainset with the additional sample
    """
    sample_to_insert_label_expended = np.expand_dims(sample_to_insert_label, 0)
    sample_to_insert_data_expended = np.expand_dims(sample_to_insert_data, 0)

    if isinstance(trainloader.dataset.train_data, torch.Tensor):
        sample_to_insert_data_expended = torch.Tensor(sample_to_insert_data_expended)

    # # Insert sample to train dataset
    dataset_train_with_sample = copy.deepcopy(trainloader.dataset)
    dataset_train_with_sample.train_data = np.concatenate((trainloader.dataset.train_data,
                                                           sample_to_insert_data_expended))
    dataset_train_with_sample.train_labels = np.concatenate((trainloader.dataset.train_labels,
                                                             sample_to_insert_label_expended))

    if isinstance(trainloader.dataset.train_data, torch.Tensor) and \
            not isinstance(dataset_train_with_sample.train_data, torch.Tensor):
        dataset_train_with_sample.train_data = \
            torch.tensor(dataset_train_with_sample.train_data,
                         dtype=trainloader.dataset.train_data.dtype)

    # Create new dataloader
    trainloader_with_sample = data.DataLoader(dataset_train_with_sample,
                                              batch_size=trainloader.batch_size,
                                              shuffle=True,
                                              num_workers=trainloader.num_workers)
    return trainloader_with_sample


# def create_svhn_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
#     """
#     create train and test pytorch dataloaders for SVHN dataset
#     :param data_dir: the folder that will contain the data
#     :param batch_size: the size of the batch for test and train loaders
#     :param num_workers: number of cpu workers which loads the GPU with the dataset
#     :return: train and test loaders along with mapping between labels and class names
#     """
#
#     trainset = datasets.CIFAR10(root=data_dir,
#                                 train=True,
#                                 download=True,
#                                 transform=transforms.Compose([transforms.ToTensor(),
#                                                               normalize_cifar]))
#     trainloader = data.DataLoader(trainset,
#                                   batch_size=batch_size,
#                                   shuffle=False,
#                                   num_workers=num_workers)
#
#     data_dir = os.path.join(data_dir, 'svhn')
#     testset = datasets.SVHN(root=data_dir,
#                             split='test',
#                             download=True,
#                             transform=transforms.Compose([transforms.ToTensor(),
#                                                           normalize_cifar]))
#
#     # Align as CIFAR10 dataset
#     testset.test_data = testset.data
#     testset.test_labels = testset.labels
#
#     testloader = data.DataLoader(testset,
#                                  batch_size=batch_size,
#                                  shuffle=False,
#                                  num_workers=num_workers)
#
#     # Classes name
#     classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     classes_svhn = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
#
#     return trainloader, testloader, classes_svhn, classes_cifar10


# class transfrom_per_img_per_ch_norm(object):
#  """ Use by calling transforms.Compose([transforms.ToTensor(), transfrom_per_img_per_ch_norm()])"
#     def __init__(self):
#         pass
#
#     def __call__(self, img):
#         num_channels = img.shape[0]
#         assert(num_channels == 3)
#         per_ch_mean = torch.zeros(num_channels)
#         img_np = img.numpy()
#         for i in range(num_channels):
#             img_np[i,:,:] = (img_np[i,:,:] - np.mean(img_np[i,:,:])) / (np.std(img_np[i,:,:]) + np.finfo(float).eps)
#
#         return torch.from_numpy(img_np)


def create_cifar100_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for CIFAR100 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR100(root=data_dir,
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               normalize_cifar]))
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              normalize_cifar]))
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
               'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
               'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
               'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
               'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
               'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
               'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
               'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
               'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
               'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
               'worm')

    return trainloader, testloader, classes


def generate_noise_sample():
    random_sample_data = np.random.randint(256, size=(32, 32, 3), dtype='uint8')
    random_sample_label = -1
    return random_sample_data, random_sample_label


def create_tensor_dataloader(input: torch.Tensor, labels: torch.Tensor, batch_size: int = 128, num_workers: int = 4,
                             start_idx: int = 0, end_idx: int = 49, labels_to_test: int = 10):
    testset = data.TensorDataset(input, labels)
    indices = [i for i in range(start_idx, end_idx+1)]
    testset = data.Subset(testset, indices=indices)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)
    classes = [str(i) for i in range(labels_to_test)]
    return testloader, classes


def create_imagenet_test_loader(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4, start_idx=0, end_idx=49, labels_to_test=1000):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    bounds = get_dataset_min_max_val("imagenet")
    imagenet_transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            # normalize,
        ])
    data_dir = data_dir + '/imagenet/val'
    testset = datasets.ImageNet(root=data_dir,
                                 split='val',
                                 download=False,
                                 transform=imagenet_transform)
    samples_per_label = end_idx - start_idx + 1
    total_samples_per_label = 50
    assert(samples_per_label <= total_samples_per_label)
    indices = [i for start_label_idx in range(0, labels_to_test*total_samples_per_label, total_samples_per_label)
               for i in range(start_label_idx, start_label_idx + samples_per_label)]
    testset = data.Subset(testset, indices=indices)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    classes = [str(i) for i in range(labels_to_test)]
    return testloader, classes, bounds


class CIFAR10Adversarial(datasets.CIFAR10):
    """
    Implementing adversarial attack to CIFAR10 testset.
    """

    def __init__(self, epsilon=0.005, adversarial_sign_dataset_path='./data/adversarial_sign', **kwargs):
        """

        :param epsilon: the strength of the attack. Fast gradient sign attack.
        :param adversarial_sign_dataset_path: path in which the gradients sign from the back propagation is saved into.
        :param kwargs: initial init arguments.
        """
        super(CIFAR10Adversarial, self).__init__(**kwargs)
        self.adversarial_sign_dataset_path = adversarial_sign_dataset_path
        self.epsilon = epsilon
        for index in range(self.test_data.shape[0]):
            sign = np.load(os.path.join(self.adversarial_sign_dataset_path, str(index) + '.npy'))
            sign = np.transpose(sign, (1, 2, 0))
            self.test_data[index] = np.clip(self.test_data[index] + (epsilon * 255) * sign, 0, 255)


class CIFAR10AdversarialTest(datasets.CIFAR10):
    """
    Implementing adversarial attack to CIFAR10 testset.
    """
    def __init__(self, attack, start_idx, end_idx, transform, **kwargs):
        """

        :param attack: the attack that will be activate on the original MNIST testset
        :param transform: use the transform (for dataset normalizations) prior to using the attack
        :param kwargs: initial init arguments for datasets.MNIST.
        """
        super(CIFAR10AdversarialTest, self).__init__(**kwargs)
        assert(self.train is False)
        assert(start_idx >= 0 and end_idx < self.data.shape[0])
        test_samples = end_idx - start_idx + 1
        grp_size = 50
        assert(test_samples % grp_size == 0)

        test_adv_data = torch.zeros([test_samples, 3, 32, 32])
        from utilities import plt_img
        # plt_img(self.data, [0])
        for index in range(test_samples):
            # use the transform on all the testset
            img = Image.fromarray(self.data[index+start_idx])
            test_adv_data[index] = transform(img)
        plt_img(test_adv_data, [0])
        self.data = test_adv_data
        self.targets = torch.LongTensor(self.targets[start_idx:end_idx+1])

        epoch_start_time = time.time()
        device = TorchUtils.get_device()
        for index in range(int(test_samples / grp_size)):
            # save the adversarial testset
            self.data[index * grp_size:(index + 1) * grp_size] = attack.create_adversarial_sample(
                self.data[index * grp_size:(index + 1) * grp_size].to(device),
                self.targets[index * grp_size:(index + 1) * grp_size].to(device)).detach()
            print("Create CIFAR adversarial testset, index: {}, time passed: {}".format(index,
                                                                                        time.time() - epoch_start_time))

        self.data = self.data.to("cpu")
        self.transform = null_transform
        plt_img(self.data, [0])

    def __getitem__(self, index):
        """
        Overwrite __getitem__  from datasets.MNIST in order to return pre-saved adversarial values
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        img = self.data[index]
        return img, target

    def __len__(self):
        return len(self.data)


def create_adversarial_cifar10_dataloaders(attack, data_dir: str = './data', batch_size: int = 128, num_workers: int = 4,
                                           start_idx=0, end_idx=9999, train_augmentation: bool = True):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset
    :param attack:
    :param data_dir: the folder that will contain the data
    :param adversarial_dir: the output dir to which the gradient adversarial sign will be saved.
    :param epsilon: the additive gradient strength to be added to the image.
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :param start_idx:
    :param end_idx:
    :return: train and test loaders along with mapping between labels and class names
    """
    bounds = get_dataset_min_max_val("cifar_adversarial")
    if train_augmentation:
        # Same transformation as in https://github.com/MadryLab/cifar10_challenge/blob/master/cifar10_input.py (Line 95)
        # No mean-std normalization were used.
        cifar_transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.nn.functional.pad(x.unsqueeze(0),
                              (4, 4, 4, 4), mode='constant', value=0).squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        cifar_transform_test = transforms.Compose([transforms.ToTensor()])
    else:
        assert(0)  # Bounds are incorrect
        cifar_transform_train = transforms.Compose([transforms.ToTensor(), normalize_cifar])
        cifar_transform_test = transforms.Compose([transforms.ToTensor(), normalize_cifar])
    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=cifar_transform_train)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train_set,
                                  num_workers=num_workers)

    testset = CIFAR10AdversarialTest(root=data_dir,
                                 train=False,
                                 download=True,
                                 transform=cifar_transform_test,
                                 attack=attack,start_idx=start_idx,
                                 end_idx=end_idx)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes, bounds


class MnistAdversarial(datasets.MNIST):
    """
    Implementing adversarial attack to CIFAR10 testset.
    """

    def __init__(self, epsilon=0.05, adversarial_sign_dataset_path='./data/mnist_adversarial_sign_batch', **kwargs):
        """

        :param epsilon: the strength of the attack. Fast gradient sign attack.
        :param adversarial_sign_dataset_path: path in which the gradients sign from the back propagation is saved into.
        :param kwargs: initial init arguments.
        """
        super(MnistAdversarial, self).__init__(**kwargs)
        self.adversarial_sign_dataset_path = adversarial_sign_dataset_path
        self.epsilon = epsilon
        self.test_data = self.test_data.type(torch.float32) #avoid underflow
        grp_size = 128
        for index in range(int(self.test_data.shape[0]/grp_size)):
            sign = np.load(os.path.join(self.adversarial_sign_dataset_path, str(index) + '.npy'))
            # sign = np.transpose(sign, (1, 2, 0)) #This is needed in cifar10 where we have 3 dimension
            sign = torch.from_numpy(sign)
            sign = sign.type(torch.float32)
            self.test_data[index*grp_size:(index+1)*grp_size] = np.clip(self.test_data[index*grp_size:(index+1)*grp_size] + (epsilon * 255) * sign, 0, 255)
        self.test_data = self.test_data.type(torch.uint8)
        assert(self.test_data.min() >= 0)


class MnistAdversarialTest(datasets.MNIST):
    """
    Implementing adversarial attack on MNIST testset.
    """

    def __init__(self, attack, start_idx, end_idx, transform, **kwargs):
        """

        :param attack: the attack that will be activate on the original MNIST testset
        :param transform: use the transform (for dataset normalizations) prior to using the attack
        :param kwargs: initial init arguments for datasets.MNIST.
        """
        super(MnistAdversarialTest, self).__init__(**kwargs)
        assert(self.train is False)
        assert(start_idx >= 0 and end_idx < self.test_data.shape[0])
        test_samples = end_idx - start_idx + 1
        grp_size = 200
        assert(test_samples % grp_size == 0)
        plt_img_list_idx = list(range(0,5))

        transform_data = torch.zeros([test_samples, 1, 28, 28])
        from utilities import plt_img
        plt_img(self.test_data, plt_img_list_idx)
        for index in range(int(test_samples)):
            # use the transform on all the testset
            img = Image.fromarray(self.test_data[index+start_idx].numpy(), mode='L')
            transform_data[index] = transform(img)

        self.adv_data = transform_data.clone()
        self.targets = torch.LongTensor(self.targets[start_idx:end_idx + 1])

        device = TorchUtils.get_device()
        epoch_start_time = time.time()
        for index in range(int(test_samples / grp_size)):
            # save the adversarial testset
            self.adv_data[index*grp_size:(index+1)*grp_size] = attack.create_adversarial_sample(
                                                self.adv_data[index*grp_size:(index+1)*grp_size].to(device),
                                                self.targets[index*grp_size:(index+1)*grp_size].to(device)).detach()
            print("Create MNIST adversarial testset, index: {}, time passed: {}".format(index,
                                                                                        time.time() - epoch_start_time))

        self.transform = null_transform
        if attack.name != 'NoAttack':
            plt_img(self.adv_data, plt_img_list_idx, True)

    def __getitem__(self, index):
        """
        Overwrite __getitem__  from datasets.MNIST in order to return pre-saved adversarial values
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        img = self.adv_data[index]
        return img, target

    def __len__(self):
        return len(self.adv_data)


def calc_norm(original_data, adv_data):
    """
    calc_norm: calculate L1 and L2 norms of ||original_data - adv_data||
    :param original_data:
    :param adv_data:
    :return:
    """
    diff = original_data - adv_data
    l2_norm = diff.view(diff.shape[0], -1).norm(p=2, dim=-1)
    l1_norm = diff.view(diff.shape[0], -1).norm(p=1, dim=-1)
    l_inf_norm = diff.view(diff.shape[0], -1).norm(p=float('inf'), dim=-1)
    print("l_inf_norm: %.3f  l2_norm: %.3f l1_norm: %.3f" % (l_inf_norm.mean(), l2_norm.mean(), l1_norm.mean()))
    return l_inf_norm, l2_norm, l1_norm


def create_adv_mnist_test_dataloader_preprocessed(attack, data_dir: str = './data', batch_size: int = 128,
                                                  num_workers: int = 4, start_idx: int = 0, end_idx: int = 9999):
    """
    Create adversarial test  dataloader for MNIST dataset. The data in the dataloader.dataset is already adversarial.
    :param attack: get instace of the adversarial attack
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    print("create_adversarial_mnist_dataloaders...")
    testset = MnistAdversarialTest(root=data_dir,
                               train=False,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             normalize_mnist]),
                               attack=attack,
                               start_idx=start_idx,
                               end_idx=end_idx)

    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return testloader, classes, get_dataset_min_max_val("mnist_adversarial")


def create_mnist_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for MNIST dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """

    trainloader, _, bounds = create_mnist_train_dataloader(data_dir, batch_size, num_workers)

    testset = datasets.MNIST(root=data_dir,
                             train=False,
                             download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           normalize_mnist]))
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return trainloader, testloader, classes, bounds


def create_mnist_train_dataloader(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for MNIST dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """

    trainset = datasets.MNIST(root=data_dir,
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            normalize_mnist]))
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train_set,
                                  num_workers=num_workers)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return trainloader, classes, get_dataset_min_max_val("mnist_adversarial")


def null_transform(x):
    """
    Overwrite the transform in datasets.MNIST (None) and return the input.
    """
    return x