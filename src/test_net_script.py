import torch
import json
import os
from experimnet_utilities import Experiment
from train_utilities import TrainClass
# import matplotlib.pyplot as plt
# from dataset_utilities import *

# trainloader, testloader, classes = create_adversarial_mnist_dataloaders(data_dir = './../data', adversarial_dir='./../data/mnist_adversarial_sign')
#
# for batch_idx, (inputs, labels) in enumerate(testloader):
#     plt.figure()
#     img = (inputs.numpy())[0,0]
#     plt.imshow(img, cmap='gray')
#     plt.show()

def train_model(experiment_type: str):
    ################
    # Load training params
    with open(os.path.join('src', 'params.json')) as f:
        params = json.load(f)

    ################
    # Class that depends ins the experiment type
    experiment_h = Experiment(experiment_type, params)
    params = experiment_h.get_params()


    ################
    # Load datasets
    data_folder = './data'
    dataloaders = experiment_h.get_dataloaders(data_folder)

    ################
    # Run basic training
    model_base = experiment_h.get_model()
    params_init_training = params['initial_training']
    train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()),
                             params_init_training['lr'],
                             params_init_training['momentum'],
                             params_init_training['step_size'],
                             params_init_training['gamma'],
                             params_init_training['weight_decay'])

    model_base.load_state_dict(torch.load(params['initial_training']['pretrained_model_path']))
    model_base = model_base.cuda() if torch.cuda.is_available() else model_base

    test_loss, test_acc = train_class.test(model_base, dataloaders['test'])
    return test_loss, test_acc


test_loss, test_acc = train_model('mnist_adversarial')
print(test_acc)
