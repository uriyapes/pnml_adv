import os
print(os.getcwd()) #print working dir
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import pandas as pd
import numpy as np
import glob
import torch
import json
import pathlib

from importlib import reload
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


def get_subdir_list(path_to_root_dir:str):
    # search_string = os.path.join(path_to_root_dir, '*').replace("\\","/")
    # search_string = path_to_root_dir + '/*'
    search_string = pathlib.Path(path_to_root_dir) / '*'
    subdir_list = glob.glob(str(search_string), recursive=False)
    return subdir_list

def load_dir_params(path:str):
    param_path = pathlib.Path(path) / 'params.json'
    param_path = glob.glob(str(param_path), recursive=True)
    assert (len(param_path) == 1)
    with open(param_path[0]) as f:
        params = json.load(f)
    return params

def load_dir_stats(path:str):
    adv = torch.load(os.path.join(path, 'adversarials.t'))
    return adv.get_accuracy(), adv.get_mean_loss()

def result_dirs_to_df(subdir_list: list, black_box_flag: bool = False):
    assert(len(subdir_list) >= 1)
    statistics_df = pd.DataFrame(columns=['acc', 'epsilon', 'mean loss', 'std loss', 'mean entropy'])
    for i, subdir in enumerate(subdir_list):
        params = load_dir_params(subdir)
        if params["adv_attack_test"]["white_box"] is False:
            path_to_blackbox_dir = os.path.dirname(params["adv_attack_test"]["black_box_adv_path"])
            attack_params = load_dir_params(path_to_blackbox_dir)
            params["adv_attack_test"] = attack_params["adv_attack_test"]
            assert(params["adv_attack_test"]["white_box"])
        acc, stats = load_dir_stats(subdir)
        statistics_df.loc[i, "acc"] = acc
        statistics_df.loc[i, "mean loss"] = stats
        statistics_df.loc[i, "epsilon"] = params["adv_attack_test"]["epsilon"]
        statistics_df.loc[i, "lambda"] = params["fit_to_sample"]["epsilon"]

    return statistics_df



# All tests were performed on the imagenet evaluation subset which includes all the samples for the first 100 classes (50 samples per each class for a total of 5000 samples).
#
# PGD attack properties: "pgd_iter": 50, "pgd_step": 0.00392156862, "pgd_test_restart_num": 10
# For 1. "epsilon": 0.0156862745098 was selected.
# For 2. lambda = 0.00692156862 (refinement strength)
# For 3. "epsilon": 0.0156862745098 was selected.; lambda = 0.00692156862 (refinement strength)

# 1.1. pNML Accuracy Vs. Refinement strength (\lambda)
natural_acc_vs_lambda_res_path = "./output/imagenet_diff_fix_natural"
pgd_acc_vs_lambda_res_path = "./output/imagenet_diff_fix_pgd"

# 1.2.  Accuracy Vs. attack strength (\epsilon); with\ without  pNML
pgd_nopnml_diff_eps_path = "./output/imagenet_pgd_diff_eps"
pgd_pnml_diff_eps_path = "./output/imagenet_pgd_diff_eps_pnml_lambda_01176"

subdir_list = get_subdir_list(pgd_acc_vs_lambda_res_path)
df_pgd = result_dirs_to_df(subdir_list)
subdir_list = get_subdir_list(natural_acc_vs_lambda_res_path)
df_natural = result_dirs_to_df(subdir_list)

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



subdir_list = get_subdir_list(pgd_nopnml_diff_eps_path)
df_pgd_nopnml = result_dirs_to_df(subdir_list)
subdir_list = get_subdir_list(pgd_pnml_diff_eps_path)
df_pgd_pnml = result_dirs_to_df(subdir_list)

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