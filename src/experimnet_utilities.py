import torch
import json
from typing import Union
from dataset_utilities import create_mnist_train_dataloader, create_adv_mnist_test_dataloader_preprocessed
from dataset_utilities import create_imagenet_test_loader
from dataset_utilities import create_adversarial_cifar10_dataloaders
from dataset_utilities import create_mnist_dataloaders
from dataset_utilities import create_tensor_dataloader
# from dataset_utilities import create_svhn_dataloaders
from models.mpl import Net, Net_800_400_100, MNISTClassifier, PnmlModel
from models.wide_resnet_original import WideResNet
from models.madry_wide_resnet import MadryWideResNet
from models.model_utils import load_pretrained_imagenet_model, load_pretrained_model, ImagenetModel
from adversarial.attacks import get_attack
from dataset_utilities import get_dataset_min_max_val


class Experiment:
    def __init__(self, args: dict, cli_params: Union[dict, None] = None):
        """

        :param args: General arguments detailing the experiment such as: output_root, path to parameters.json (param_file_path)
                    and experiment_type
        :param cli_params:
        """
        if args['experiment_type'] not in [
                            'out_of_dist_svhn',
                            'out_of_dist_noise',
                            'pnml_mnist',
                            'imagenet_adversarial',
                            'cifar_adversarial',
                            'mnist_adversarial']:
            raise NameError('No experiment type: %s' % type)
        self.exp_type = args['experiment_type']
        self.params = self.__load_params_from_file(args, self.exp_type)
        if self.exp_type != "imagenet_adversarial":
            self.params["num_classes"] = 10
        if cli_params is None:
            cli_params = dict()
        self.__update_params_from_cli(cli_params)
        self.output_dir = args['output_root']

    @staticmethod
    def __load_params_from_file(args, exp_type):
        """
        Load parameters for exp_type
        :param args: a dict containing a path to parameters file containing parameters for different experiments
        :param exp_type:
        :return:
        """
        param_file_path = args['param_file_path']
        with open(param_file_path) as f:         # Load the params for all experiments from param_file_path
            params = json.load(f)
        assert(exp_type == params['exp_type'])
        return params

    def __update_params_from_cli(self, cli_params):
        for key, inner_dict in cli_params.items():
            for inner_key, val in inner_dict.items():
                if val is not None:
                    print("Update: params[{}][{}] = {}".format(key, inner_key, val))
                    self.params[key][inner_key] = val

    def get_params(self):
        return self.params

    def get_dataloaders(self) -> dict:
        """
        :return: Non adversarial dataloaders
        """
        if self.params['adv_attack_test']["white_box"]:
            return self.get_adv_dataloaders(datafolder='./data', p=None, model=None)
        else:
            return self.get_blackbox_dataloader()

    def get_blackbox_dataloader(self):
        p = self.params['adv_attack_test']
        assert(p["white_box"] is False)
        adv = torch.load(p["black_box_adv_path"])
        dataloader = dict()
        dataloader['test'], dataloader['classes'] = create_tensor_dataloader(adv.adversarial_sample, adv.true_label,
                                                             batch_size=self.params["batch_size"], num_workers=self.params["num_workers"],
                                                             start_idx=p["test_start_idx"], end_idx=p["test_end_idx"])
        dataloader['dataset_name'] = self.exp_type
        dataloader["black_box_attack_params"] = adv.attack_params
        return dataloader

    def get_adv_dataloaders(self, datafolder: str = './data', p=None, model=None):
        """
        :param datafolder: location of the data
        :param p: (dict) the adversarial attack parameters
        :param model: the black/white-box model on which the attack will work, if None no attack will run
        :return: dataloaders dict
        """
        if p is None:
            p = {'attack_type': "no_attack"}
        if model is None or p['attack_type'] == "no_attack":
            attack = get_attack(p)
        else:
            model.eval()
            attack = get_attack(p, model, get_dataset_min_max_val(self.exp_type))
        return self._create_dataloaders(datafolder, attack)

    def _create_dataloaders(self, data_folder: str = './data', attack=None):
        if self.exp_type == 'pnml_mnist':
            trainloader, testloader, classes, bounds = create_mnist_dataloaders(data_folder,
                                                                        self.params['batch_size'],
                                                                        self.params['num_workers'])
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'classes': classes,
                           'bounds': bounds}
        # elif self.exp_type == 'out_of_dist_svhn':
        #     trainloader, testloader_svhn, classes_svhn, classes_cifar10 = create_svhn_dataloaders(data_folder,
        #                                                                                           self.params[
        #                                                                                               'batch_size'],
        #                                                                                           self.params[
        #                                                                                               'num_workers'])
        #     dataloaders = {'train': trainloader,
        #                    'test': testloader_svhn,
        #                    'classes': classes_cifar10,
        #                    'classes_svhn': classes_svhn}
        elif self.exp_type == 'imagenet_adversarial':
            assert (attack is not None)
            testloader, classes, bounds = create_imagenet_test_loader(data_folder,
                                                              self.params['batch_size'], self.params['num_workers'],
                                                              self.params['adv_attack_test']['test_start_idx'],
                                                              self.params['adv_attack_test']['test_end_idx'],
                                                              self.params["num_classes"]
                                                              )
            dataloaders = {'test': testloader,
                           'classes': classes,
                           'bounds': bounds}
        elif self.exp_type == 'cifar_adversarial':
            assert(attack is not None)
            trainloader, testloader, classes, bounds = create_adversarial_cifar10_dataloaders(attack, data_folder,
                                                                    self.params['batch_size'], self.params['num_workers'],
                                                                    self.params['adv_attack_test']['test_start_idx'],
                                                                    self.params['adv_attack_test']['test_end_idx'])
            adv_test_flag = True if attack.name != "NoAttack" else False  # This flag indicates whether the testset is already adversarial
            dataloaders = {'train': trainloader,
                           'test': testloader,
                           'adv_test_flag': adv_test_flag,  # This flag indicates whether the testset is already adversarial
                           'classes': classes, 'bounds': bounds}

        elif self.exp_type == 'mnist_adversarial':
            assert(attack is not None)
            dataloaders = dict()
            dataloaders['train'], dataloaders['classes'], bounds_train = create_mnist_train_dataloader(data_folder,
                                                                 self.params['batch_size'], self.params['num_workers'])

            dataloaders['adv_test_flag'] = True if attack.name != "NoAttack" else False # This flag indicates whether the testset is already adversarial
            dataloaders['test'], _, bounds_test = create_adv_mnist_test_dataloader_preprocessed(attack, data_folder,
                                                                 self.params['batch_size'], self.params['num_workers'],
                                                                 self.params['adv_attack_test']['test_start_idx'],
                                                                  self.params['adv_attack_test']['test_end_idx'])
            assert(bounds_train == bounds_test)
            dataloaders['bounds'] = bounds_train
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        dataloaders['dataset_name'] = self.exp_type
        return dataloaders

    def get_model(self, model_arch: str, ckpt_path: str, pnml_model_flag: bool = False, pnml_model_keep_grad: bool =True):
        """
        Load a untrained or trained model according to the experiment type and if a ckpt_path is given.
        :param model_arch: the architecture of the model
        :param ckpt_path: the path to the model .ckpt file. If no ckpt_path is given then the initial model is loaded
        :param pnml_model_flag: If true, return PnmlModel of the loaded model
        :return: A NN model
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
        elif self.exp_type == "cifar_adversarial":
            if model_arch == 'wide_resnet':
                model = MadryWideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
            elif model_arch == "RST": # Model used in "Unlabeled Data Improves Adversarial Robustness" paper, which has the same architecture as in the original WideResNet
                model = WideResNet(depth=28, num_classes=10, widen_factor=10)
            else:
                raise NameError('No model_arch type %s for %s experiment' % (str(model_arch), self.exp_type))
        elif self.exp_type == "imagenet_adversarial":
            if model_arch == 'resnet50':
                model = load_pretrained_imagenet_model("resnet50")
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        model = load_pretrained_model(model, ckpt_path) if ckpt_path is not None else model
        model = ImagenetModel(model, self.params["num_classes"]) if self.exp_type == "imagenet_adversarial" else model
        if pnml_model_flag:
            model = PnmlModel(model, self.params['fit_to_sample'], get_dataset_min_max_val(self.exp_type), self.params["num_classes"], pnml_model_keep_grad)
        model.ckpt_path = ckpt_path
        return model

    def get_exp_name(self):
        if self.exp_type == 'out_of_dist_svhn':
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

    def get_attack_for_model(self, model):
        return get_attack(self.params["adv_attack_test"], model, get_dataset_min_max_val(self.exp_type))
