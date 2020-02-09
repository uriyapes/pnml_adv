import json

from dataset_utilities import create_mnist_train_dataloader, create_adv_mnist_test_dataloader_preprocessed
from dataset_utilities import create_imagenet_test_loader
from dataset_utilities import create_adversarial_cifar10_dataloaders
from dataset_utilities import create_cifar10_dataloaders
from dataset_utilities import create_cifar10_random_label_dataloaders
from dataset_utilities import create_mnist_dataloaders
from dataset_utilities import create_svhn_dataloaders
from dataset_utilities import dataloaders_noise
from models.mpl import Net, Net_800_400_100, MNISTClassifier, PnmlModel
from models.resnet import resnet20, load_pretrained_resnet20_cifar10_model
from models.wide_resnet_original import WideResNet
from models.wide_resnet import WideResNet as MadryWideResNet
from models.model_utils import load_pretrained_imagenet_model, load_pretrained_model
from adversarial.attacks import get_attack
from dataset_utilities import get_dataset_min_max_val


class Experiment:
    def __init__(self, args):

        if args['experiment_type'] not in ['pnml_cifar10',
                            'random_labels',
                            'out_of_dist_svhn',
                            'out_of_dist_noise',
                            'pnml_mnist',
                            'imagenet_adversarial',
                            'cifar_adversarial',
                            'mnist_adversarial']:
            raise NameError('No experiment type: %s' % type)
        self.exp_type = args['experiment_type']
        self.params = self.__load_params_from_file(args, self.exp_type)
        self.output_dir = args['output_root']

    @staticmethod
    def __load_params_from_file(args, exp_type):
        param_file_path = args['param_file_path']
        with open(param_file_path) as f:         # Load the params for all experiments from param_file_path
            params = json.load(f)
        params = params[exp_type]
        # Overwrite params from arguments given
        if args['first_idx'] is not None and args['last_idx'] is not None:
            params['adv_attack_test']['test_start_idx'] = args['first_idx']
            params['adv_attack_test']['test_end_idx'] = args['last_idx']
        if args['test_eps'] is not None:
            params['adv_attack_test']['epsilon'] = args['test_eps']
        if args['test_step_size'] is not None:
            params['adv_attack_test']['pgd_step'] = args['test_step_size']
        if args['test_pgd_iter'] is not None:
            params['adv_attack_test']['pgd_iter'] = args['test_pgd_iter']
        if args['beta'] is not None:
            params['adv_attack_test']['beta'] = args['beta']

        if args['lambda'] is not None:
            params['fit_to_sample']['epsilon'] = args['lambda']
        if args['fix_pgd_iter'] is not None:
            params['fit_to_sample']['pgd_iter'] = args['fix_pgd_iter']
            params['fit_to_sample']['pgd_step'] = params['fit_to_sample']['epsilon'] / args['fix_pgd_iter']
        if args['fix_pgd_restart_num'] is not None:
            params['fit_to_sample']['pgd_rand_start'] = True if args['fix_pgd_restart_num'] != 0 else False
            params['fit_to_sample']['pgd_test_restart_num'] = args['fix_pgd_restart_num'] if params['fit_to_sample']['pgd_rand_start'] else 1

        return params

    def get_params(self):
        return self.params

    def get_adv_dataloaders(self, datafolder, p, model=None):
        """
        :param experiment_h: experiment class instance
        :param datafolder: location of the data
        :param p: (dict) the adversarial attack parameters
        :param model: the black/white-box model on which the attack will work, if None no attack will run
        :return: dataloaders dict
        """
        if model is None or p['attack_type'] == "no_attack":
            attack = get_attack("no_attack")
        else:
            model.eval()
            attack = get_attack(p['attack_type'], model, p['epsilon'], p['pgd_iter'], p['pgd_step'],
                                p['pgd_rand_start'], get_dataset_min_max_val(self.exp_type), p['pgd_test_restart_num'],
                                beta=p['beta'])
        return self.get_dataloaders(datafolder, attack)

    def get_dataloaders(self, data_folder: str = './data', attack=None):
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
        elif self.exp_type == 'imagenet_adversarial':
            assert (attack is not None)
            testloader, classes = create_imagenet_test_loader(data_folder,
                                                              self.params['batch_size'], self.params['num_workers'])
            dataloaders = {'test':testloader,
                           'classes': classes}
        elif self.exp_type == 'cifar_adversarial':
            assert(attack is not None)
            trainloader, testloader, classes = create_adversarial_cifar10_dataloaders(attack, data_folder,
                                                                    self.params['batch_size'], self.params['num_workers'],
                                                                    self.params['adv_attack_test']['test_start_idx'],
                                                                    self.params['adv_attack_test']['test_end_idx'])
            adv_test_flag = True if attack.name is not "NoAttack" else False  # This flag indicates whether the testset is already adversarial
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'adv_test_flag': adv_test_flag,  # This flag indicates whether the testset is already adversarial
                           'classes': classes
                            }

        elif self.exp_type == 'mnist_adversarial':
            assert(attack is not None)
            dataloaders = dict()
            dataloaders['train'], dataloaders['classes'] = create_mnist_train_dataloader(data_folder,
                                                                 self.params['batch_size'], self.params['num_workers'])

            dataloaders['adv_test_flag'] = True if attack.name is not "NoAttack" else False # This flag indicates whether the testset is already adversarial
            dataloaders['test'], _ = create_adv_mnist_test_dataloader_preprocessed(attack, data_folder,
                                                                 self.params['batch_size'], self.params['num_workers'],
                                                                 self.params['adv_attack_test']['test_start_idx'],
                                                                  self.params['adv_attack_test']['test_end_idx'])
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        dataloaders['dataset_name'] = self.exp_type
        return dataloaders

    def get_model(self, model_arch: str, ckpt_path: str):
        """
        Load a untrained or trained model according to the experiment type and if a ckpt_path is given.
        :param model_arch: the architecture of the model
        :param ckpt_path: the path to the model .ckpt file. If no ckpt_path is given then the initial model is loaded
        :return:
        """
        ckpt_path = None if ckpt_path == "None" else ckpt_path
        if self.exp_type == "mnist_adversarial":
            if model_arch == 'Net':
                model = Net()
            elif model_arch == 'Net_800_400_100':
                model = Net_800_400_100()
            elif model_arch == 'MNISTClassifier':
                model = MNISTClassifier()
            elif model_arch == 'PnmlModel':
                model = MNISTClassifier()
            else:
                raise NameError('No model_arch type %s for %s experiment' % (str(model_arch), self.exp_type))
            model = load_pretrained_model(model, ckpt_path) if ckpt_path is not None else model

            if model_arch == 'PnmlModel':
                model = PnmlModel(model, self.params['fit_to_sample'], get_dataset_min_max_val(self.exp_type))

        elif self.exp_type == "cifar_adversarial":
            if model_arch == 'wide_resnet':
                model = MadryWideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
            elif model_arch == "RST" or model_arch == "PnmlModel": # Model used in "Unlabeled Data Improves Adversarial Robustness" paper, which has the same architecture as in the original WideResNet
                model = WideResNet(depth=28, num_classes=10, widen_factor=10)
            else:
                raise NameError('No model_arch type %s for %s experiment' % (str(model_arch), self.exp_type))
            model = load_pretrained_model(model, ckpt_path) if ckpt_path is not None else model
            if model_arch == "PnmlModel":
                model = PnmlModel(model, self.params['fit_to_sample'], get_dataset_min_max_val(self.exp_type))
        elif self.exp_type == "imagenet_adversarial":
            model = load_pretrained_imagenet_model(model_arch)
        elif self.exp_type == 'pnml_cifar10':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
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
        elif self.exp_type == 'imagenet_adversarial':
            name = 'imagenet_adversarial'
        elif self.exp_type == 'cifar_adversarial':
            name = 'cifar_adversarial'
        elif self.exp_type == 'mnist_adversarial':
            name = 'mnist_adversarial'
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        return name
