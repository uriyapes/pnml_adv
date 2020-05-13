import jsonargparse
import os
import torch
import time

import logger_utilities
from experimnet_utilities import Experiment
from utilities import TorchUtils
TorchUtils.set_rnd_seed(1)
# Uncomment for performance. Comment for debug and reproducibility
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
from train_utilities import TrainClass
import json


def eval_adversarial_dataset(model, dataloader, attack, save_adv_sample: bool =False):
    """
    Evaluate model performance on dataloader with attack
    :param model:
    :param dataloader:
    :param attack:
    :param save_adv_sample:
    :return:
    """
    logger = logger_utilities.get_logger()
    model.eval()
    loss = 0
    correct = 0
    adversarials = None
    logger.info("Starting eval...")
    adv_batch_l = []
    for iter_num, (data, labels) in enumerate(dataloader):

        t0 = time.time()
        data, labels = TorchUtils.to_device(data), TorchUtils.to_device(labels)
        adversarials_batch = attack.create_adversarial_sample(data, labels, get_adversarial_class=True,
                                                              save_adv_sample=save_adv_sample)
        t1 = time.time()
        adv_batch_l.append(adversarials_batch)
        logger.info("eval_model iter_num: {} ,process time: {} ".format(iter_num, t1 - t0))
        # if iter_num == 0:
        #     adversarials = adversarials_batch
        # else:
        #     adversarials.merge(adversarials_batch)
        # t2 = time.time()
        # logger.info("eval_model iter_num: {} ,process time: {} save time: {} ".format(iter_num, t1-t0, t2-t1))

    adversarials = adv_batch_l[0].cat(adv_batch_l)
    return adversarials



def generate_adv_dataset(model, dataloader, attack):
    pass


def eval_black_box(model_to_eval, model_to_attack, dataloader, attack):
    pass


def main():
    parser = jsonargparse.ArgumentParser(description='General arguments', default_meta=False)
    parser.add_argument('-t', '--general.experiment_type', default='imagenet_adversarial',
                        help='Type of experiment to execute', type=str)
    parser.add_argument('-p', '--general.param_file_path', default=os.path.join('./src/parameters', 'eval_imagenet_param.json'),
                        help='param file path used to load the parameters file containing default values to all '
                             'parameters', type=str)
    # parser.add_argument('-p', '--general.param_file_path', default='src/tests/test_mnist_pgd_with_pnml_expected_result/params.json',
    #                     help='param file path used to load the parameters file containing default values to all '
    #                          'parameters', type=str)
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
    logger_utilities.init_logger(logger_name=exp.get_exp_name(), output_root=exp.output_dir)
    logger = logger_utilities.get_logger()
    # Get models:
    model_to_eval = exp.get_model(exp.params['model']['model_arch'], exp.params['model']['ckpt_path'],
                                  exp.params['model']['pnml_active'], True if exp.params["adv_attack_test"]["attack_type"] != "natural" else False)

    dataloaders = exp.get_dataloaders()

    # Get adversarial attack:
    attack = exp.get_attack_for_model(model_to_eval)


    with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(exp.params, indent=4, sort_keys=False))
    logger.info(exp.params)

    adv = eval_adversarial_dataset(model_to_eval, dataloaders['test'], attack)
    loss = adv.get_mean_loss()
    acc = adv.get_accuracy()
    logger.info("Accuracy: {}, Loss: {}".format(acc, loss))
    adv.dump(logger.output_folder)
    return adv


if __name__ == "__main__":
    adv = main()
    print('Finish evaluation')