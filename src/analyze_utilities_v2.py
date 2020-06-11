import os
print(os.getcwd()) #print working dir
import matplotlib.pyplot as plt
import time
import pandas as pd
import glob
import torch
import json
import pathlib
from typing import Union

os.sys.path.insert(0, '../src/')

output_path = './results/imagenet'
is_plot_title = False
is_save_fig = True

# print(plt.style.available)
# print(plt.rcParams)
plt.style.use(['seaborn-darkgrid', 'seaborn-paper'])
label_size = 18
tick_size = 14
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('axes', titlesize=label_size)
plt.rc('axes', labelsize=label_size)
plt.rc('xtick', labelsize=tick_size)
plt.rc('ytick', labelsize=tick_size)


def get_subdir_list(path_to_root_dir: str):
    # search_string = pathlib.Path(path_to_root_dir) / '*'
    # subdir_list = glob.glob(str(search_string), recursive=False)
    p = pathlib.Path(path_to_root_dir)
    subdir_list = [d for d in p.iterdir() if d.is_dir()]
    return subdir_list


def merge_splitted_results_from_subdir(subdir_list: list):
    """
    Extract all adversarials.t from path sub directories, assert that the parameters are the same and unify them
    :param subdir_list: The path to the directory containing subdir that should be unified.
    :return: A unified adversarials object.
    """
    assert(len(subdir_list) > 1)
    params1 = load_dir_params(subdir_list[0])
    # Make sure all subdir contains the same configuration besides the indices:
    for i, subdir in enumerate(subdir_list[1:]):
        params2 = load_dir_params(subdir)
        assert(is_dicts_equal(params1["adv_attack_test"], params2["adv_attack_test"], ["test_start_idx", "test_end_idx"]))
        assert(is_dicts_equal(params1["model"], params2["model"]))
        assert(is_dicts_equal(params1["fit_to_sample"], params2["fit_to_sample"]))

    adv_l = []
    for i, subdir in enumerate(subdir_list):
        adv_l.append(torch.load(os.path.join(subdir, 'adversarials.t')))
    return adv_l[0].cat(adv_l), params1


def is_dicts_equal(d1: dict, d2: dict, ignore_keys: Union[None, list] = None):
    if ignore_keys is None:
        ignore_keys = []
    return {k: v for k, v in d1.items() if k not in ignore_keys} == {k: v for k, v in d2.items() if k not in ignore_keys}


def load_dir_params(path: str):
    param_path = pathlib.Path(path) / 'params.json'
    if param_path.exists() is False:
        param_path = '..' / pathlib.Path(path) / 'params.json'
    param_path_l = glob.glob(str(param_path), recursive=False)
    assert len(param_path_l) == 1, "for param_path: {} found the following param files: {}".format(param_path, param_path_l)
    with open(param_path_l[0]) as f:
        params = json.load(f)

    if params["adv_attack_test"]["white_box"] is False:
        path_to_blackbox_dir = os.path.dirname(params["adv_attack_test"]["black_box_adv_path"])
        attack_params = load_dir_params(path_to_blackbox_dir)
        params["adv_attack_test"] = attack_params["adv_attack_test"]
        assert (params["adv_attack_test"]["white_box"])
    return params


def load_exp_result_from_dir(root_dir: str, indices: Union[list, None]):
    """
    Load adversarials results and param dict of an experiment. Results are contained in the root dir or inside multiple
     subdir.
    :param root_dir: The path to the root directory containing the results
    :param indices: if not None, include only results of indices
    :return: adversarial object and a param dict
    """
    subdir_list = get_subdir_list(root_dir)
    if(len(subdir_list) > 1):
        adv, params = merge_splitted_results_from_subdir(subdir_list)
    else:
        if len(subdir_list) == 1:
            root_dir = subdir_list[0]
        try:
            adv = torch.load(os.path.join(root_dir, 'adversarials.t'))
        except FileNotFoundError:
            adv = torch.load(os.path.join(root_dir, 'adversarials_compress.t'))

        params = load_dir_params(root_dir)

    if indices is not None:
        adv.correct = adv.correct[indices]
        adv.loss = adv.loss[indices]
    return adv, params


def compress_and_save_adv(subdir_list: list):
    """
    Save compressed adversarial object, without image data.
    :param subdir_list: a list of subdirectories for which the function will save a compressed version of adversarials.t
    """
    for i, subdir in enumerate(subdir_list):
        adv, params = load_exp_result_from_dir(subdir, None)
        if adv.original_sample is not None or adv.adversarial_sample is not None:
            print("Compress adversarial.t in directory: {}".format(subdir))
            adv.original_sample = None
            adv.adversarial_sample = None
            adv.dump(subdir, "adversarials_compress.t")


def results_dirs_to_df(subdir_list: list, indices: Union[list, None] = None) -> pd.DataFrame:
    """
    Load multiple experiments that are stored in subdir_list and save them into a DF.
    :param subdir_list: a list of folders containing the results
    :param indices: if not None, include only results of indices
    :return: A dataframe containing the results.
    """
    assert(len(subdir_list) >= 1)
    statistics_df = pd.DataFrame(columns=['acc', 'epsilon', 'mean loss', 'samples'])
    for i, subdir in enumerate(subdir_list):
        # params = load_dir_params(subdir)
        # if params["adv_attack_test"]["white_box"] is False:
        #     path_to_blackbox_dir = os.path.dirname(params["adv_attack_test"]["black_box_adv_path"])
        #     attack_params = load_dir_params(path_to_blackbox_dir)
        #     params["adv_attack_test"] = attack_params["adv_attack_test"]
        #     assert(params["adv_attack_test"]["white_box"])
        adv, params = load_exp_result_from_dir(subdir, indices)
        statistics_df.loc[i, "acc"] = adv.get_accuracy()
        statistics_df.loc[i, "mean loss"] = adv.get_mean_loss()
        statistics_df.loc[i, "mean size"] = adv.get_mean_loss()
        statistics_df.loc[i, "samples"] = len(adv.correct)
        if params["adv_attack_test"]['attack_type'] == 'natural':
            statistics_df.loc[i, "epsilon"] = 0
            statistics_df.loc[i, "iter"] = 0
            statistics_df.loc[i, "pgd_step"] = 0
            statistics_df.loc[i, "restarts"] = 0
        else:
            statistics_df.loc[i, "epsilon"] = params["adv_attack_test"]["epsilon"]
            statistics_df.loc[i, "iter"] = params["adv_attack_test"]["pgd_iter"]
            statistics_df.loc[i, "pgd_step"] = params["adv_attack_test"]["pgd_step"]
            statistics_df.loc[i, "restarts"] = params["adv_attack_test"]["pgd_test_restart_num"]
        statistics_df.loc[i, "lambda"] = params["fit_to_sample"]["epsilon"]

    return statistics_df


if __name__ == "__main__":
    print("start")
    # All tests were performed on the imagenet evaluation subset which includes all the samples for the first 100 classes (50 samples per each class for a total of 5000 samples).
    #
    # Experiments:
    # 1. pNML Accuracy Vs. Refinement strength (\lambda) - Shows the trade-off between natural and robust accuracy
    # 2. Accuracy Vs. PGD-attack strength (\epsilon); with\ without  pNML - Shows that pNML improve accuracy.
    # 3. Accuracy Vs. Adaptive-attack & PGD-attack strength (\epsilon) - Shows that adaptive attack is not working
    # 4. TODO: blackbox attack Accuracy Vs. Blackbox-attack for different epsilons strength (\epsilon) - Shows that pNML improve accuracy compared to blackbox without pNML
    # PGD attack properties: "pgd_iter": 50, "pgd_step": 0.00392156862, "pgd_test_restart_num": 10
    # For 1. "epsilon": 0.0156862745098 was selected.
    # For 2. lambda = 0.01176 (refinement strength)
    # For 3. "epsilon": [0.0156, 0.0313, 0.0627] was selected.; lambda = 0.01176(refinement strength). indices: 1 per label

    # 1.1. pNML Accuracy Vs. Refinement strength (\lambda)
    natural_acc_vs_lambda_res_path = "./output/imagenet_diff_fix_natural"
    pgd_acc_vs_lambda_res_path = "./output/imagenet_diff_fix_pgd"

    # 1.2.  Accuracy Vs. attack strength (\epsilon); with\ without  pNML
    pgd_nopnml_diff_eps_path = "./output/imagenet_pgd_diff_eps"
    pgd_pnml_diff_eps_path = "./output/imagenet_pgd_diff_eps_pnml_lambda_01176"

    # 1.3 Accuracy Vs. Adaptive-attack & PGD-attack strength (\epsilon)
    adaptive_diff_eps_path = "./output/imagenet_adaptive"


    # 1.1
    subdir_list = get_subdir_list(pgd_acc_vs_lambda_res_path)
    df_pgd = results_dirs_to_df(subdir_list)
    subdir_list = get_subdir_list(natural_acc_vs_lambda_res_path)
    df_natural = results_dirs_to_df(subdir_list)

    df_natural = df_natural.sort_values("lambda", ignore_index=True)
    df_pgd = df_pgd.sort_values("lambda", ignore_index=True)
    axes = plt.gca()
    # axes.set_xlim([0.0, 0.12])
    # axes.set_ylim([0.0, 1.0])
    plt.figure(0)
    plt.plot(df_natural['lambda'], df_natural['acc'], 'go-', label='Natural')
    plt.plot(df_pgd['lambda'], df_pgd['acc'], 'bo-', label='PGD')
    plt.legend(fontsize=12, ncol=1, loc=1)
    plt.xlabel('Refinement strength')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_path, 'Natural_vs_refinement_strength.jpg'), dpi=200,
                bbox_inches=plt.tight_layout()) if is_save_fig else None
    plt.show()


    # 1.2
    subdir_list = get_subdir_list(pgd_nopnml_diff_eps_path)
    df_pgd_nopnml = results_dirs_to_df(subdir_list)
    subdir_list = get_subdir_list(pgd_pnml_diff_eps_path)
    df_pgd_pnml = results_dirs_to_df(subdir_list)

    df_pgd_nopnml = df_pgd_nopnml.sort_values("epsilon", ignore_index=True)
    df_pgd_pnml = df_pgd_pnml.sort_values("epsilon", ignore_index=True)
    axes = plt.gca()
    # axes.set_xlim([0.0, 0.12])
    # axes.set_ylim([0.0, 1.0])
    plt.figure(1)
    plt.plot(df_pgd_nopnml['epsilon'], df_pgd_nopnml['acc'], 'go--', label='no pNML')
    plt.plot(df_pgd_pnml['epsilon'], df_pgd_pnml['acc'], 'bo--', label='pNML')
    plt.legend(fontsize=12, ncol=1, loc=1)
    plt.xlabel('Attack strength')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_path, 'pgd_pnml_and_nopnml.jpg'), dpi=200,
                bbox_inches=plt.tight_layout()) if is_save_fig else None
    plt.show()

    # 1.3
    indices = [i for i in range(0, 4999+1, 50)]
    subdir_list = get_subdir_list(adaptive_diff_eps_path)
    df_adaptive = results_dirs_to_df(subdir_list)
    df_adaptive = df_adaptive.sort_values("epsilon", ignore_index=True)
    subdir_list = get_subdir_list(pgd_pnml_diff_eps_path)
    df_pgd_pnml = results_dirs_to_df(subdir_list, indices)

    axes = plt.gca()
    plt.figure(2)
    plt.plot(df_adaptive['epsilon'], df_adaptive['acc'], 'go--', label='Adaptive')
    plt.plot(df_pgd_pnml['epsilon'], df_pgd_pnml['acc'], 'bo--', label='PGD')
    plt.legend(fontsize=12, ncol=1, loc=1)
    plt.xlabel('Attack strength')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_path, 'adaptive_vs_pgd.jpg'), dpi=200,
                bbox_inches=plt.tight_layout()) if is_save_fig else None
    plt.show()