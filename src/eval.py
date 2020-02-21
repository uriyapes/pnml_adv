import torch
import argparse
from deepfool.foolbox import foolbox
import numpy as np
import os
import pandas as pd
from logger_utilities import Logger
from experimnet_utilities import Experiment
from dataset_utilities import get_dataset_min_max_val
from utilities import TorchUtils
import logging
import time
# TorchUtils.set_device("cpu")

def eval_hopskipjump(model, dataloader, iterations=2, logger=None):
    logger.info("Starting eval_hopskipjump with {} iterations".format(iterations))
    low_bound, upper_bound = get_dataset_min_max_val("mnist_adversarial", dtype=np.float32)
    fmodel = foolbox.models.PyTorchModel(model, bounds=(float(low_bound), float(upper_bound)), num_classes=10)
    # images, labels = foolbox.utils.samples(dataset='mnist', batchsize=1, data_format='channels_first',
    #                                        bounds=(0, 1))

    attack = foolbox.attacks.HopSkipJumpAttack(fmodel, distance=foolbox.distances.Linf)
    adv_l = []
    t1 = time.time()
    for iter_num, (images, labels) in enumerate(dataloader):
        logger.info("eval_hopskipjump batch number: {}. Time passed: {}".format(iter_num, (time.time()-t1)))
        t1 = time.time()
        images, labels = images.numpy(), labels.numpy()

        # model(torch.Tensor(images))
        # images = np.expand_dims(images, axis=1)
        adversarials_batch = attack(images, labels, iterations=iterations, unpack=False, log_every_n_steps=None, loggingLevel=logging.WARNING)
        # print(adversarials_batch[0].distance)
        adv_l = adv_l + adversarials_batch
        adv_l_repack, _ = repack_adversarial_results(adv_l)
        logger.dump_pickle(adv_l_repack)

        # if iter_num == 1:
            # break
    return adv_l


def unpack_adv_obj(adv_l, img_shape):
    none_img = np.ones(img_shape) * np.nan
    advs = [a['perturbed'] for a in adv_l]
    advs = [p if p is not None else none_img for p in advs]
    advs = np.stack(advs)
    return advs


def calc_statistics(model, adv_l, logger):
    """
    Calculate statistics from from adv_l
    :param adv_l: a repacked adversarial list (see repack_adversarial_results func)
    """
    ### Calculate attack unsuccessful rate ###
    # Evaluate
    index = 0
    batch_size = 32
    correct = 0
    while index < len(adv_l):
        batch = adv_l[index:(index + batch_size)]
        data_batch = torch.Tensor(unpack_adv_obj(batch, adv_l[0]['perturbed'].shape))
        data_batch = TorchUtils.to_device(data_batch)
        labels_batch = np.stack([a['original_class'] for a in batch])
        outputs = model(data_batch)
        _, predicted = torch.max(outputs.detach_().data, 1)
        # print(predicted)
        # print(labels_batch)
        correct += (predicted.cpu().numpy() == labels_batch).sum().item()
        index += batch_size

    unsuccessful_attack = correct / len(adv_l)
    logger.info("Unsuccessful attacks {}%".format(100 * unsuccessful_attack))
    # print(np.mean(model.forward(adv_img).argmax(axis=-1) == labels))
    ### - END - ###

    ### Calculate median distance ###
    distances = np.asarray([a["distance"] for a in adv_l])
    logger.info("Distance - min:{:.1e}, median:{:.1e}, max:{:.1e}".format(distances.min(), np.median(distances), distances.max()))
    logger.info("{} of {} attacks failed".format(sum(a["distance"] == np.inf for a in adv_l), len(adv_l)))
    logger.info("{} of {} inputs misclassified without perturbation".format(
        sum(a["distance"] == 0 for a in adv_l), len(adv_l)))

    # Present number of queries for the first 10 samples
    for i in range(10):
        print("Number of queries for {} sample: {}".format(i, adv_l[i]['queries']))


def repack_adversarial_results(adv_l):
    """
    # Repack for serialization - ignore unnecessary unpicklized attributes
    :param adv_l: a list containing adversarials objects as defined in foolbox
    :return: a list of dict, each containing the relevant data of the adversarial object
    """
    adv_l_pickle = []
    res_df = pd.DataFrame(columns=["index", "queries", "best_query_call", "distance",
                                   "adversarial_class", "original_class"])

    for index, adv in enumerate(adv_l):
        res_df.loc[index] = [index, adv._total_prediction_calls, adv._best_prediction_calls, adv.distance.value,
                         adv.adversarial_class, adv.original_class]
        adv_l_pickle.append({"index": index, "perturbed": adv.perturbed, "queries": adv._total_prediction_calls,
                             "best_query_call": adv._best_prediction_calls, "output": adv.output, "distance": adv.distance.value,
                             "adversarial_class": adv.adversarial_class, "original_class": adv.original_class})
    return adv_l_pickle, res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applications of Deep PNML')
    parser.add_argument('-t', '--experiment_type', default='mnist_adversarial',
                        help='Type of experiment to execute', type=str)
    parser.add_argument('-f', '--first_idx', default=None, help='first test idx', type=int)
    parser.add_argument('-l', '--last_idx', default=None, help='last test idx', type=int)
    parser.add_argument('-p', '--param_file_path', default=os.path.join('src', 'params.json'),
                        help='param file path used to load the parameters file containing default values to all '
                             'parameters', type=str)
    parser.add_argument('-e', '--test_eps', default=None, help='the epsilon strength of the attack', type=float)
    parser.add_argument('-ts', '--test_step_size', default=None, help='the step size of the attack', type=float)
    parser.add_argument('-ti', '--test_pgd_iter', default=None, help='the number of test pgd iterations', type=int)
    parser.add_argument('-r', '--lambda', default=None, help='the epsilon strength of the refinement (lambda)', type=float)
    parser.add_argument('-b', '--beta', default=None, help='the beta value for regret reduction regularization ', type=float)
    parser.add_argument('-i', '--fix_pgd_iter', default=None, help='the number of PGD iterations of the refinement', type=int)
    parser.add_argument('-n', '--fix_pgd_restart_num', default=None, help='the number of PGD restarts where 0 means no random start',
                        type=int)
    parser.add_argument('-o', '--output_root', default='output', help='the output directory where results will be saved', type=str)
    args = vars(parser.parse_args())
    experiment_h = Experiment(args)
    model = experiment_h.get_model("PnmlModel", "./trained_models/mnist_classifier/bpda_ep6_eps0.3_restant20_uniformRnd/model_iter_6.pt")
    model.eval()
    dataloaders = experiment_h.get_adv_dataloaders()

    ################
    # Create logger and save params to output folder
    logger = Logger(experiment_type=experiment_h.get_exp_name(), output_root=experiment_h.output_dir)
    # logger = Logger(experiment_type='TMP', output_root='output')
    logger.info('OutputDirectory: %s' % logger.output_folder)
    logger.info('Device: %s' % TorchUtils.get_device())
    # with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
    #     outfile.write(json.dumps(experiment_h.get_params(), indent=4, sort_keys=False))
    logger.info(experiment_h.get_params())

    adv_l = eval_hopskipjump(model, dataloaders['test'], 50, logger)
    adv_l_repack, res_df = repack_adversarial_results(adv_l)
    logger.dump_pickle(adv_l_repack)
    res_df.to_pickle(os.path.join(logger.output_folder, 'res_df.pkl'))
    calc_statistics(model, adv_l_repack, logger)
    # pd.read_pickle(os.path.join(logger.output_folder, 'res_df.pkl'))
