import torch
import jsonargparse
from deepfool.foolbox import foolbox
import numpy as np
import os
import sys
import pandas as pd
import logger_utilities
import logging
from experimnet_utilities import Experiment
from dataset_utilities import get_dataset_min_max_val
from utilities import TorchUtils
import time
import multiprocessing as mp
import glob
import pickle

TorchUtils.set_rnd_seed(1)
# Uncomment for performance. Comment for debug and reproducibility
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def eval_batch(images: np.ndarray, labels: np.ndarray, iterations: int, batch_iteration: int, exp: Experiment, logger: logger_utilities.Logger, cuda_dev: int):
    # with torch.cuda.set_device(cuda_dev):
    logger.info("Start run batch {} in cuda_device {}".format(batch_iteration, cuda_dev))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev)

    # logger.info("Create model...")
        # model = exp.get_model("PnmlModel", "./trained_models/mnist_classifier/bpda_ep6_eps0.3_restant20_uniformRnd/model_iter_6.pt")
    model_to_eval = exp.get_model(exp.params['model']['model_arch'], exp.params['model']['ckpt_path'],
                                  exp.params['model']['pnml_active'], False)
    model_to_eval.eval()
    model_to_eval.freeze_all_layers()
    logger.info("Create foolbox model and attack...")
    # low_bound, upper_bound = get_dataset_min_max_val("mnist_adversarial", dtype=np.float32)
    low_bound, upper_bound = get_dataset_min_max_val(exp.exp_type, dtype=np.float32)
    fmodel = foolbox.models.PyTorchModel(model_to_eval, bounds=(float(low_bound), float(upper_bound)), num_classes=exp.params["num_classes"])
    attack = foolbox.attacks.HopSkipJumpAttack(fmodel, distance=foolbox.distances.Linf)
    adversarials_batch = attack(images, labels, iterations=iterations, unpack=False, log_every_n_steps=1,
                                loggingLevel=logging.INFO, batch_size=exp.params["batch_size"])

    adversarials_batch_repack, _ = repack_adversarial_results(adversarials_batch)
    logger.dump_pickle(adversarials_batch_repack, "adv_batch_%.2d.p" % (batch_iteration))
    logger.info("Finish run batch {} in cuda_device {}".format(batch_iteration, cuda_dev))
    sys.stdout.flush()


class CudaPoll:
    def __init__(self, num_cuda_dev: int, logger):
        self.num_cuda_dev = num_cuda_dev
        self.poll = {str(i): True for i in range(self.num_cuda_dev)}  # Init all available
        self.logger = logger

    def get_free_dev(self):
        for i in range(self.num_cuda_dev):
            if self.poll[str(i)] is True:
                return i
        return -1

    def alloc_dev(self, proc: mp.Process, dev_num: int):
        assert(self.poll[str(dev_num)])
        self.poll[str(dev_num)] = proc

    def check_dev_status(self):
        for i in range(self.num_cuda_dev):
            if self.poll[str(i)] is not True:
                p = self.poll[str(i)]
                if p.is_alive() is False:
                    if p.exitcode != 0:
                        self.logger.info("Error, in device {} process exit with exitcode {} ".format(i, p.exitcode))
                    self.poll[str(i)] = True


def eval_dataset(dataloader, iterations, logger, experiment_h):
    logger.info("Starting eval_hopskipjump with {} iterations".format(iterations))
    t1 = time.time()
    timeout = 1000000
    num_gpu = 16
    cuda_poll = CudaPoll(num_gpu, logger)
    # proc_l = []
    cuda_dev = 0
    for batch_iteration, (images, labels) in enumerate(dataloader):
        images, labels = images.numpy(), labels.numpy()
        while True:
            cuda_poll.check_dev_status()
            cuda_dev = cuda_poll.get_free_dev()
            if cuda_dev != -1: break
            else: time.sleep(10)

        p = mp.Process(target=eval_batch, args=(images, labels, iterations, batch_iteration, experiment_h, logger, cuda_dev))
        cuda_poll.alloc_dev(p, cuda_dev)
        p.start()
        logger.info("Start run batch {} in cuda_device {} time {}".format(batch_iteration, cuda_dev, (time.time()-t1)))
        time.sleep(10)
        # proc_l.append(p)
        # if (batch_iteration+1) % num_gpu == 0:
        #     cuda_dev = 0
        #     i = 0
        #     while len(proc_l) != 0:
        #         p = proc_l.pop()
        #         p.join()
        #         p.is_alive()
        #         if p.exitcode != 0:
        #             logger.info("Error, during batch {} process exit with exitcode {} after time: {}".format(batch_iteration-i, p.exitcode, (time.time() - t1)))
        #         i += 1
        #     torch.cuda.empty_cache()
        #     time.sleep(6)  # Give GC enough time


def unpack_adv_obj(adv_l: list) -> np.ndarray:
    """
    Get a list of repacked adversarials and a
    :param adv_l: a repacked adversarial list (see repack_adversarial_results func)
    :return: a numpy array of stack images
    """
    img_shape = adv_l[0]['perturbed'].shape
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
        data_batch = torch.Tensor(unpack_adv_obj(batch))
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

def load_pickles(path_to_folder:str) -> list:
    """
    Search for all pickles files, load them and insert them into a list in order.
    :param path_to_folder: A path to result folder
    :return: A list of repacked adversarials.
    """
    pathname = path_to_folder + '/*.p'
    files_l = sorted(glob.glob(pathname,  recursive=True))
    adv_l_repack = []
    for f in files_l:
        l = pickle.load(open(f, "rb"))
        adv_l_repack = adv_l_repack + l
    return adv_l_repack

def adv_results_repacked_to_df(adv_l_repack) -> pd.DataFrame:
    """
    Get adv_l_repack and return a corresponding pandas dataframe with the relevant statistics
    :param adv_l_repack: the output of repack_adversarial_results - a list of dict, each containing the relevant data of
                            the adversarial object
    :return: DF containing the stats in the list
    """
    keys = ["index", "queries", "best_query_call", "distance", "adversarial_class", "original_class"]
    res_df = pd.DataFrame(columns=keys)
    for index, adv in enumerate(adv_l_repack):
        res_df.loc[index] = [adv[k] for k in keys]
    return res_df


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


def _main():
    mp.set_start_method('spawn')
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
    parser.add_argument('--adv_attack_test.attack_type', help='attack type', type=str, default="natural")
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

    experiment_h = Experiment(general_args, args)
    dataloaders = experiment_h.get_adv_dataloaders()
    ################
    # Create logger and save params to output folder
    logger_utilities.init_logger(logger_name=experiment_h.get_exp_name(), output_root=experiment_h.output_dir)
    logger = logger_utilities.get_logger()
    # logger = Logger(experiment_type='TMP', output_root='output')
    logger.info('OutputDirectory: %s' % logger.output_folder)
    logger.info('Device: %s' % TorchUtils.get_device())
    logger.info(experiment_h.get_params())
    eval_dataset(dataloaders['test'], 29, logger, experiment_h)
    logger.info("Done")


if __name__ == "__main__":
    _main()
    exit()
    #
    # parser = argparse.ArgumentParser(description='Applications of Deep PNML')
    # parser.add_argument('-t', '--experiment_type', default='mnist_adversarial',
    #                     help='Type of experiment to execute', type=str)
    # parser.add_argument('-f', '--first_idx', default=None, help='first test idx', type=int)
    # parser.add_argument('-l', '--last_idx', default=None, help='last test idx', type=int)
    # parser.add_argument('-p', '--param_file_path', default=os.path.join('src', 'params.json'),
    #                     help='param file path used to load the parameters file containing default values to all '
    #                          'parameters', type=str)
    # parser.add_argument('-e', '--test_eps', default=None, help='the epsilon strength of the attack', type=float)
    # parser.add_argument('-ts', '--test_step_size', default=None, help='the step size of the attack', type=float)
    # parser.add_argument('-ti', '--test_pgd_iter', default=None, help='the number of test pgd iterations', type=int)
    # parser.add_argument('-r', '--lambda', default=None, help='the epsilon strength of the refinement (lambda)', type=float)
    # parser.add_argument('-b', '--beta', default=None, help='the beta value for regret reduction regularization ', type=float)
    # parser.add_argument('-i', '--fix_pgd_iter', default=None, help='the number of PGD iterations of the refinement', type=int)
    # parser.add_argument('-n', '--fix_pgd_restart_num', default=None, help='the number of PGD restarts where 0 means no random start',
    #                     type=int)
    # parser.add_argument('-o', '--output_root', default='output', help='the output directory where results will be saved', type=str)
    # args = vars(parser.parse_args())
    # experiment_h = Experiment(args)
    # model = experiment_h.get_model("PnmlModel", "./trained_models/mnist_classifier/bpda_ep6_eps0.3_restant20_uniformRnd/model_iter_6.pt")
    # model.eval()
    # model.freeze_all_layers()
    # dataloaders = experiment_h.get_adv_dataloaders()
    #
    # ################
    # # Create logger and save params to output folder
    # logger = Logger(experiment_type=experiment_h.get_exp_name(), output_root=experiment_h.output_dir)
    # # logger = Logger(experiment_type='TMP', output_root='output')
    # logger.info('OutputDirectory: %s' % logger.output_folder)
    # logger.info('Device: %s' % TorchUtils.get_device())
    # # with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
    # #     outfile.write(json.dumps(experiment_h.get_params(), indent=4, sort_keys=False))
    # logger.info(experiment_h.get_params())
    #
    # adv_l = eval_hopskipjump(model, dataloaders['test'], 50, logger, experiment_h)
    # adv_l_repack, res_df = repack_adversarial_results(adv_l)
    # logger.dump_pickle(adv_l_repack)
    # res_df.to_pickle(os.path.join(logger.output_folder, 'res_df.pkl'))
    # calc_statistics(model, adv_l_repack, logger)
    # # pd.read_pickle(os.path.join(logger.output_folder, 'res_df.pkl'))
