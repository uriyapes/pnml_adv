import torch
import json
import os
from experimnet_utilities import Experiment
from train_utilities import TrainClass
from logger_utilities import Logger
import matplotlib.pyplot as plt
from dataset_utilities import *

print(os.getcwd()) #print working dir
def plt_adv_mnist_img():
    trainloader, testloader, classes = create_adversarial_mnist_dataloaders(data_dir='./data', adversarial_dir='./data/mnist_adversarial_sign_batch', epsilon=0.25)

    num_of_img_to_plt = 10
    for batch_idx, (inputs, labels) in enumerate(testloader):
        i = 0
        for i in range(min(inputs.shape[0], num_of_img_to_plt)):
            plt.figure()
            img = (inputs.numpy())[i,0]
            plt.imshow(img, cmap='gray')
            plt.show()
        break



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
    # Create logger and save params to output folder
    logger = Logger(experiment_type=experiment_h.get_exp_name(), output_root='output_temp')
    # logger = Logger(experiment_type='TMP', output_root='output')
    logger.info('OutputDirectory: %s' % logger.output_folder)
    with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))
    logger.info(params)

    ################
    # Load datasets
    data_folder = './data'
    adv_data_folder = os.path.join('data', 'mnist_adversarial_sign_batch')
    dataloaders = experiment_h.get_dataloaders(data_folder, adv_data_folder)

    ################
    # Run basic training
    model_base = experiment_h.get_model()
    params_init_training = params['initial_training']
    train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()),
                             params_init_training['lr'],
                             params_init_training['momentum'],
                             params_init_training['step_size'],
                             params_init_training['gamma'],
                             params_init_training['weight_decay'],
                             logger,
                             params_init_training["adv_alpha"], params["epsilon"])

    if params_init_training["do_initial_training"]:
        model_base, train_loss, test_loss = train_class.train_model(model_base, dataloaders,
                                                                   params_init_training['epochs'])
    else:
        ##load model and test it
        model_base.load_state_dict(torch.load(params['initial_training']['pretrained_model_path']))
        model_base = model_base.cuda() if torch.cuda.is_available() else model_base

    test_loss, test_acc = train_class.test(model_base, dataloaders['test'])
    return test_loss, test_acc, model_base


test_loss, test_acc, model_trained = train_model('mnist_adversarial')
print(test_acc)
