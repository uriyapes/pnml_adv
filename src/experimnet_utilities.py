import os

from dataset_utilities import create_mnist_train_dataloader, create_adv_mnist_test_dataloader
from dataset_utilities import create_adversarial_cifar10_dataloaders
from dataset_utilities import create_cifar10_dataloaders
from dataset_utilities import create_cifar10_random_label_dataloaders
from dataset_utilities import create_mnist_dataloaders
from dataset_utilities import create_svhn_dataloaders
from dataset_utilities import dataloaders_noise
from adversarial_utilities import create_adversarial_mnist_sign_dataset
from mpl import Net, Net_800_400_100, MNISTClassifier, load_pretrained_model
from resnet import resnet20, resnet56, resnet110, load_pretrained_resnet20_cifar10_model
from wide_resnet_original import WideResNet
from models.wide_resnet import WideResNet as MadryWideResNet


class Experiment:
    def __init__(self, exp_type: str, params: dict, first_idx, last_idx):
        if exp_type not in ['pnml_cifar10',
                            'random_labels',
                            'out_of_dist_svhn',
                            'out_of_dist_noise',
                            'pnml_mnist',
                            'cifar_adversarial',
                            'mnist_adversarial']:
            raise NameError('No experiment type: %s' % type)
        self.params = params
        self.exp_type = exp_type
        self.executed_get_params = False
        self.first_idx=first_idx
        self.last_idx = last_idx

    def get_params(self):
        if self.exp_type == 'pnml_cifar10':
            self.params = self.params['pnml_cifar10']
        elif self.exp_type == 'random_labels':
            self.params = self.params['random_labels']
        elif self.exp_type == 'out_of_dist_svhn':
            self.params = self.params['pnml_cifar10']
        elif self.exp_type == 'out_of_dist_noise':
            self.params = self.params['pnml_cifar10']
        elif self.exp_type == 'pnml_mnist':
            self.params = self.params['pnml_mnist']
        elif self.exp_type == 'cifar_adversarial':
            self.params = self.params['cifar_adversarial']
        elif self.exp_type == 'mnist_adversarial':
            self.params = self.params['mnist_adversarial']
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        if self.first_idx is not None and self.last_idx is not None:
            # Overwrite params
            self.params['test_start_idx'] = self.first_idx
            self.params['test_end_idx'] = self.last_idx

        self.executed_get_params = True
        return self.params

    def get_dataloaders(self, data_folder: str = './data', testset_black_box_attack=None):

        if self.executed_get_params is False:
            _ = self.get_params()

        if self.exp_type == 'pnml_cifar10':
            trainloader, testloader, classes = create_cifar10_dataloaders(data_folder,
                                                                          self.params['batch_size'],
                                                                          self.params['num_workers'])
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes}
        elif self.exp_type == 'random_labels':
            trainloader, testloader, classes = create_cifar10_random_label_dataloaders(data_folder,
                                                                                       self.params['batch_size'],
                                                                                       self.params['num_workers'],
                                                                                       label_corrupt_prob=self.params[
                                                                                           'label_corrupt_prob'])
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes}
        elif self.exp_type == 'out_of_dist_svhn':
            trainloader, testloader_svhn, classes_svhn, classes_cifar10 = create_svhn_dataloaders(data_folder,
                                                                                                  self.params[
                                                                                                      'batch_size'],
                                                                                                  self.params[
                                                                                                      'num_workers'])
            dataloaders = {'train': trainloader,
                           'test': testloader_svhn,
                           'classes': classes_cifar10,
                           'classes_svhn': classes_svhn}

        elif self.exp_type == 'out_of_dist_noise':
            dataloaders = dataloaders_noise(data_folder,
                                            self.params['batch_size'],
                                            self.params['num_workers'])

        elif self.exp_type == 'pnml_mnist':
            trainloader, testloader, classes = create_mnist_dataloaders(data_folder,
                                                                        self.params['batch_size'],
                                                                        self.params['num_workers'])
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes}
        elif self.exp_type == 'cifar_adversarial':
            trainloader, testloader, classes = create_cifar10_dataloaders(data_folder, self.params['batch_size'],
                                                                          self.params['num_workers'])
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes}
        elif self.exp_type == 'mnist_adversarial':
            dataloaders = {}
            trainloader, classes = create_mnist_train_dataloader(data_folder, self.params['batch_size'],
                                                                 self.params['num_workers'])
            dataloaders['train'] = trainloader
            dataloaders['classes'] = classes
            if testset_black_box_attack is not None:
                testloader, _ = create_adv_mnist_test_dataloader(testset_black_box_attack, data_folder, self.params['batch_size'], self.params['num_workers'])
                dataloaders['test'] = testloader
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        return dataloaders

    def get_model(self, model_arch: str = None):

        if self.exp_type == 'pnml_cifar10':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'random_labels':
            model = WideResNet()
        elif self.exp_type == 'out_of_dist_svhn':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'out_of_dist_noise':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'pnml_mnist':
            model = Net()
        elif self.exp_type == 'cifar_adversarial':
            # model = load_pretrained_resnet20_cifar10_model(resnet20())
            # model = resnet110()
            # model = WideResNet()
            model = MadryWideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
        elif self.exp_type == 'mnist_adversarial':
            if model_arch == 'Net':
                model = Net()
            elif model_arch == 'Net_800_400_100':
                model = Net_800_400_100()
            elif model_arch == 'MNISTClassifier':
                model = MNISTClassifier()
            else:
                raise NameError('No model_arch type %s' % str(model_arch))
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        return model

    def get_exp_name(self):

        if self.exp_type == 'pnml_cifar10':
            name = 'pnml_cifar10'
        elif self.exp_type == 'random_labels':
            name = 'random_labels'
        elif self.exp_type == 'out_of_dist_svhn':
            name = 'out_of_dist_svhn'
        elif self.exp_type == 'out_of_dist_noise':
            name = 'out_of_dist_noise'
        elif self.exp_type == 'pnml_mnist':
            name = 'pnml_mnist'
        elif self.exp_type == 'cifar_adversarial':
            name = 'cifar_adversarial'
        elif self.exp_type == 'mnist_adversarial':
            name = 'mnist_adversarial'
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        return name
