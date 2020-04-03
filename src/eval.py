import jsonargparse
import os

import torch
torch.manual_seed(1)
import numpy as np
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(0)

import logger_utilities
from experimnet_utilities import Experiment
from utilities import TorchUtils
from train_utilities import TrainClass
import json


def eval_batch(model, data, labels):
    pass


def eval_adversarial_dataset(model, dataloader, attack):
    """
    Evaluate model performance on dataloader with attack
    :param model:
    :param dataloader:
    :param attack:
    :return:
    """
    logger = logger_utilities.get_logger()
    model.eval()
    loss = 0
    correct = 0
    adversarials = None
    for iter_num, (data, labels) in enumerate(dataloader):
        logger.info("eval_model iter_num: {}".format(iter_num))

        data, labels = TorchUtils.to_device(data), TorchUtils.to_device(labels)
        adversarials_batch = attack.create_adversarial_sample(data, labels, get_adversarial_class=True)
        if iter_num == 0:
            adversarials = adversarials_batch
        else:
            adversarials.merge(adversarials_batch)
        # outputs, batch_loss = TrainClass.__forward_pass(model, data, labels, loss_func)
        #
        # loss += float(batch_loss.detach_()) * len(data)  # loss sum for all the batch
        # _, predicted = torch.max(outputs.detach_().data, 1)
        # correct += (predicted == labels).sum().item()
        # if (predicted == labels).sum().item() == 1:
        #     print("correct prediction in iter_num: {}".format(iter_num))
    loss = adversarials.loss.sum() / len(dataloader.dataset)
    acc = adversarials.correct.sum().item() / len(dataloader.dataset)
    logger.info("Accuracy: {}, Loss: {}".format(acc, loss))
    adversarials.dump(logger.output_folder)
    return adversarials



def generate_adv_dataset(model, dataloader, attack):
    pass


def eval_black_box(model_to_eval, model_to_attack, dataloader, attack):
    pass


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description='General arguments', default_meta=False)
    parser.add_argument('-t', '--general.experiment_type', default='mnist_adversarial',
                        help='Type of experiment to execute', type=str)
    parser.add_argument('-p', '--general.param_file_path', default=os.path.join('./src/parameters', 'eval_imagenet_param.json'),
                        help='param file path used to load the parameters file containing default values to all '
                             'parameters', type=str)
    parser.add_argument('-o', '--general.output_root', default='output', help='the output directory where results will be saved', type=str)

    parser.add_argument('-f', '--adv_attack_test.test_start_idx', help='first test idx', type=int)
    parser.add_argument('-l', '--adv_attack_test.test_end_idx', help='last test idx', type=int)
    parser.add_argument('-e', '--adv_attack_test.epsilon', help='the epsilon strength of the attack', type=float)
    parser.add_argument('-ts', '--adv_attack_test.pgd_step', help='the step size of the attack', type=float)
    parser.add_argument('-ti', '--adv_attack_test.pgd_iter', help='the number of test pgd iterations', type=int)
    parser.add_argument('-b', '--adv_attack_test.beta', help='the beta value for regret reduction regularization ',
                               type=float)
    parser.add_argument('-r', '--fit_to_sample.epsilon', help='the epsilon strength of the refinement (lambda)', type=float)
    parser.add_argument('-i', '--fit_to_sample.pgd_iter', help='the number of PGD iterations of the refinement', type=int)
    parser.add_argument('-s', '--fit_to_sample.pgd_step', help='the step size of the refinement', type=float)
    parser.add_argument('-n', '--fit_to_sample.pgd_test_restart_num', help='the number of PGD restarts where 0 means no random start',
                        type=int)

    args = jsonargparse.namespace_to_dict(parser.parse_args())
    general_args = args.pop('general')

    exp = Experiment(general_args, args)
    logger_utilities.init_logger(experiment_type=exp.get_exp_name(), output_root=exp.output_dir)

    # Get models:
    model_to_eval = exp.get_model(exp.params['model']['model_arch'], exp.params['model']['ckpt_path'],
                                  exp.params['model']['pnml_active'])
    if exp.params['adv_attack_test']['white_box'] is False:
        blackbox_model = exp.get_model(exp.params['adv_attack_test']['black_box_model_arch'],
                                       exp.params['adv_attack_test']['black_box_model_path'], exp.params['adv_attack_test']['black_box_pnml_active'])

    # Get Dataloaders
    data_folder = "./data"
    dataloaders = exp.get_adv_dataloaders(data_folder)

    # Get adversarial attack:
    attack = exp.get_attack_for_model(model_to_eval)

    logger = logger_utilities.get_logger()
    with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(exp.params, indent=4, sort_keys=False))
    logger.info(exp.params)

    adv = eval_adversarial_dataset(model_to_eval, dataloaders['test'], attack)

    print('Finish evaluation')
