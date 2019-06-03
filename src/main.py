import copy
import json
import os
import time

import argparse
import torch
torch.manual_seed(1)
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


from experimnet_utilities import Experiment
from logger_utilities import Logger
from train_utilities import TrainClass, eval_single_sample, execute_pnml_training
from train_utilities import freeze_model_layers
from mpl import load_pretrained_model
from adversarial.attacks import get_attack
from dataset_utilities import mnist_max_val, mnist_min_val

"""
Example of running:
CUDA_VISIBLE_DEVICES=0 python src/main.py -t pnml_cifar10
"""


def run_experiment(experiment_type: str, first_idx: int = None, last_idx: int = None):
    ################
    # Load training params
    with open(os.path.join('src', 'params.json')) as f:
        params = json.load(f)

    ################
    # Class that depends ins the experiment type
    experiment_h = Experiment(experiment_type, params, first_idx, last_idx)
    params = experiment_h.get_params()

    ################
    # Create logger and save params to output folder
    logger = Logger(experiment_type=experiment_h.get_exp_name(), output_root='output')
    # logger = Logger(experiment_type='TMP', output_root='output')
    logger.info('OutputDirectory: %s' % logger.output_folder)
    with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))
    logger.info(params)

    ################
    # Load datasets
    data_folder = './data'
    logger.info('Load datasets: %s' % data_folder)

    # Create a black-box attack for the testset using the model from black_box_model_path
    if params['adv_attack_test']['white_box'] is False:
        p = params['adv_attack_test']
        model = load_pretrained_model(experiment_h.get_model(p['black_box_model_arch']), p['black_box_model_path'])
        attack = get_attack(p['attack_type'], model, p['epsilon'], p['pgd_iter'], p['pgd_step'],
                            p['pgd_rand_start'], (mnist_min_val, mnist_max_val))
        dataloaders = experiment_h.get_dataloaders(data_folder, attack)
    else:
        # The testset is based on whitebox attack, therefore no need to create the testset in advance
        dataloaders = experiment_h.get_dataloaders(data_folder, None)



    ################
    # Run basic training- so the base model will be in the same conditions as NML model
    params_init_training = params['initial_training']
    model_base = experiment_h.get_model(params['initial_training']['model_arch'])
    if 'initial_training' in params and params['initial_training']['do_initial_training'] is True:
        logger.info('Execute basic training')
        train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()),
                                 params_init_training['lr'],
                                 params_init_training['momentum'],
                                 params_init_training['step_size'],
                                 params_init_training['gamma'],
                                 params_init_training['weight_decay'],
                                 logger.logger,
                                 params_init_training["adv_alpha"], params_init_training["epsilon"],
                                 params_init_training["attack_type"], params_init_training["pgd_iter"],
                                 params_init_training["pgd_step"]) # TODO: not all experiments contain adversarial parameters, fix this
        train_class.eval_test_during_train = False
        train_class.freeze_batch_norm = False
        acc_goal = params_init_training['acc_goal'] if 'acc_goal' in params_init_training else None
        model_base, train_loss, test_loss = train_class.train_model(model_base, dataloaders,
                                                                    params_init_training['epochs'], acc_goal)
        torch.save(model_base.state_dict(),
                   os.path.join(logger.output_folder, '%s_model_%f.pt' % (experiment_h.get_exp_name(), train_loss)))
    elif 'initial_training' in params and params['initial_training']['do_initial_training'] is False:
        logger.info('Load pretrained model')
        model_base = load_pretrained_model(model_base, params['initial_training']['pretrained_model_path'])

    # Create a white-box attack and use it to create the testset
    if params['adv_attack_test']['white_box'] is True:
        p = params['adv_attack_test']
        attack = get_attack(p['attack_type'], model_base, p['epsilon'], p['pgd_iter'], p['pgd_step'],
                            p['pgd_rand_start'], (mnist_min_val, mnist_max_val))
        dataloaders = experiment_h.get_dataloaders(data_folder, attack)




    ################
    # Freeze layers
    logger.info('Freeze layer: %d' % params['freeze_layer'])
    model_base = freeze_model_layers(model_base, params['freeze_layer'], logger)

    ############################
    # Train ERM model to be as same as PNML
    logger.info('Train ERM model')
    params_fit_to_sample = params['fit_to_sample']
    model_erm = copy.deepcopy(model_base)
    train_class = TrainClass(filter(lambda p: p.requires_grad, model_erm.parameters()),
                             params_fit_to_sample['lr'],
                             params_fit_to_sample['momentum'],
                             params_fit_to_sample['step_size'],
                             params_fit_to_sample['gamma'],
                             params_fit_to_sample['weight_decay'],
                             logger.logger,
                             params_init_training["adv_alpha"], params_init_training["epsilon"],
                             params_init_training["attack_type"], params_init_training["pgd_iter"],
                             params_init_training["pgd_step"])
    base_train_loss, base_train_acc = train_class.eval(model_base, dataloaders['train'])
    base_test_loss, base_test_acc = train_class.eval(model_base, dataloaders['test'])
    logger.info('Base model ----- [train test] loss =[%f %f] acc=[%f %f]' %
                (base_train_loss, base_test_loss, base_train_acc, base_test_acc))

    train_class.eval_test_during_train = True
    model_erm, train_loss, test_loss = train_class.train_model(model_erm, dataloaders,
                                                               params_fit_to_sample['epochs'])
    ############################
    # Iterate over test dataset
    logger.info('Execute PNML')
    logger.info('Iterate over test dataset')
    for idx in range(params_fit_to_sample['test_start_idx'], params_fit_to_sample['test_end_idx'] + 1):
        time_start_idx = time.time()

        # Extract a sample from test dataset and check output of base model
        # sample_test_data = dataloaders['test'].dataset.test_data[idx]
        # sample_test_true_label = dataloaders['test'].dataset.test_labels[idx]
        # sample_test_data_trans = sample_test_data
        sample_test_data_trans, sample_test_true_label = dataloaders['test'].dataset[idx]
        # # Make sure the data is HxWxC:
        # if len(sample_test_data.shape) == 3 and sample_test_data.shape[2] > sample_test_data.shape[0]:
        #     sample_test_data = sample_test_data.transpose([1, 2, 0])
        #
        # # Execute transformation
        # sample_test_data_for_trans = copy.deepcopy(sample_test_data)
        # if len(sample_test_data.shape) == 2:
        #     sample_test_data_for_trans = sample_test_data_for_trans.unsqueeze(2).numpy()
        # sample_test_data_trans = dataloaders['test'].dataset.transform(sample_test_data_for_trans)

        # Evaluate with base model
        prob_org, _ = eval_single_sample(model_erm, sample_test_data_trans)
        logger.add_org_prob_to_results_dict(idx, prob_org, sample_test_true_label)

        # NML training- train the model with test sample
        execute_pnml_training(params_fit_to_sample, params_init_training, dataloaders, sample_test_data_trans, sample_test_true_label, idx,
                              model_base, logger, genie_only_training=False, adv_train=False)

        # Log and save
        logger.save_json_file()
        time_idx = time.time() - time_start_idx
        logger.info('----- Finish %s idx = %d, time=%f[sec] ----' % (experiment_h.get_exp_name(), idx, time_idx))
    logger.info('Finish All!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applications of Deep PNML')
    parser.add_argument('-t', '--experiment_type', default='mnist_adversarial',
                        help='Type of experiment to execute',
                        type=str)
    parser.add_argument('-f', '--first_idx', default=None, help='first test idx', type=int)
    parser.add_argument('-l', '--last_idx', default=None, help='last test idx', type=int)
    args = vars(parser.parse_args())

    # Available experiment_type:
    #   'pnml_cifar10'
    #   'random_labels'
    #   'out_of_dist_svhn'
    #   'out_of_dist_noise'
    #   'pnml_mnist'

    run_experiment(args['experiment_type'], args['first_idx'], args['last_idx'])
    print('Finish experiment')
