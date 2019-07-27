import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import entropy
import glob

from dataset_utilities import create_cifar10_dataloaders

# Import tesloader for random label experiment
# _, testloader, _ = create_cifar10_dataloaders('../data/', 1, 1)


def extract_probabilities_list(evaluation_dict):
    # if the sample was trained with label 2, extract the prob to be 2 ...
    # return list of probabilities
    prob_all = []

    true_label = evaluation_dict['true_label'] if 'true_label' in evaluation_dict else None
    prob_org = np.array(evaluation_dict['original']['prob'])
    for trained_label in evaluation_dict:

        # One of the key is a string, ignore it
        if trained_label.isdigit():
            prob_on_trained = evaluation_dict[trained_label]['prob'][int(trained_label)]
            prob_all.append(prob_on_trained)
    predicted_label = np.argmax(prob_all) if len(prob_all) > 0 else None

    return np.array(prob_all), true_label, predicted_label, prob_org


def extract_genie_probabilities_list(evaluation_dict):
    # Extract to probabilities of the model which was trained with the true label
    # return list of probabilities
    true_label = evaluation_dict['true_label']
    prob_genie = np.array(evaluation_dict[str(true_label)]['prob'])
    predicted_genie_label = np.argmax(prob_genie)

    return prob_genie, true_label, predicted_genie_label


def execute_normalize_prob(prob_list):
    # Normalize the probabilities to be valid distribution
    # Return list of probabilities along with the normalization factor which was used.
    normalization_factor = np.sum(prob_list)
    normalized_prob = np.array(prob_list) / normalization_factor
    return normalized_prob, normalization_factor


def compute_log_loss(normalized_prob, true_label):
    # Compute the log loss
    return -np.log10(normalized_prob[true_label] + np.finfo(float).eps)


def calculate_top_k_acc(results_dict, top_k, prob_thresh=0.0):
    is_correct_nml_list = []
    is_correct_erm_list = []
    test_sample_idx_list = []
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob, true_label, predicted_label, prob_org = extract_probabilities_list(sample_dict)
        normalized_prob, _ = execute_normalize_prob(prob)

        top_k_labels = np.argsort(normalized_prob)[-top_k:][::-1].astype(int)
        if true_label in top_k_labels and normalized_prob[true_label] > prob_thresh:
            is_correct_nml_list.append(True)
        else:
            is_correct_nml_list.append(False)

        top_k_labels = np.argsort(prob_org)[-top_k:][::-1].astype(int)
        if true_label in top_k_labels and prob_org[true_label] > prob_thresh:
            is_correct_erm_list.append(True)
        else:
            is_correct_erm_list.append(False)

        test_sample_idx_list.append(keys)
    acc_top_k_nml = np.sum(is_correct_nml_list) / len(is_correct_nml_list)
    acc_top_k_erm = np.sum(is_correct_erm_list) / len(is_correct_erm_list)

    return acc_top_k_nml, acc_top_k_erm


def load_dict_from_file_list(files):
    result_dict = {}
    for file in files:
        with open(file) as f:
            result_dict.update(json.load(f))
    return result_dict


def load_results_to_df(files, is_random_labels=False, is_out_of_dist=False, idx=None):
    results_dict = load_dict_from_file_list(files)

    # NML
    pnml_df = result_dict_to_nml_df(results_dict, is_random_labels=is_random_labels, is_out_of_dist=is_out_of_dist)
    if idx is None: # if not idx is given (empty set) then take all the indexes
        idx = pnml_df.index.values
    else:
        idx = set(idx)
    statistic_pnml_df = calc_statistic_from_df_single(pnml_df.loc[idx]).rename(columns={'statistics': 'nml'})
    pnml_df = pnml_df.add_prefix('nml_')
    pnml_df = pnml_df.rename(columns={'nml_log10_norm_factor': 'log10_norm_factor'})

    # ERM
    erm_df = result_dict_to_erm_df(results_dict, is_random_labels=is_random_labels, is_out_of_dist=is_out_of_dist)
    statistic_erm_df = calc_statistic_from_df_single(erm_df.loc[idx]).rename(columns={'statistics': 'erm'})
    erm_df = erm_df.add_prefix('erm_')

    # genie
    genie_df, statistic_genie_df = None, None
    if is_out_of_dist is False:
        genie_df = result_dict_to_genie_df(results_dict, is_random_labels=is_random_labels)
        statistic_genie_df = calc_statistic_from_df_single(genie_df.loc[idx]).rename(columns={'statistics': 'genie'})
        genie_df = genie_df.add_prefix('genie_')

    # Merge and return
    result_df = pd.concat([pnml_df, erm_df, genie_df], axis=1)
    statistic_df = pd.concat([statistic_pnml_df, statistic_erm_df, statistic_genie_df], axis=1)
    return result_df, statistic_df


def calc_statistic_from_df_single(result_df):
    mean_loss, std_loss = result_df['loss'].mean(), result_df['loss'].std()
    acc = result_df['is_correct'].sum() / result_df.shape[0]
    mean_entropy = np.mean(result_df['entropy'])
    statistics_df = pd.DataFrame(
        {'statistics': pd.Series([acc, mean_loss, std_loss, mean_entropy],
                                 index=['acc', 'mean loss', 'std loss', 'mean entropy'])})
    return statistics_df


def result_dict_to_nml_df(results_dict, is_random_labels=False, is_out_of_dist=False):
    # Initialize col of df
    cls_keys = list(filter(lambda l: l.isdigit(), list(results_dict['0'].keys())))  # extract only the digits keys
    cls_keys = [int(k) for k in cls_keys]
    df_col = [str(x) for x in range(min(cls_keys), max(cls_keys)+1)] + \
             ['true_label', 'loss', 'log10_norm_factor', 'entropy']
    nml_dict = {}
    for col in df_col:
        nml_dict[col] = []
    loc = []

    # Iterate on test samples
    for keys in results_dict:
        sample_dict = results_dict[keys]
        prob_all, true_label, predicted_label, _ = extract_probabilities_list(sample_dict)
        prob_nml, norm_factor = execute_normalize_prob(prob_all)
        true_label = true_label if is_random_labels is False else testloader.dataset.test_labels[int(keys)]
        nml_dict['true_label'].append(true_label)
        nml_dict['loss'].append(compute_log_loss(prob_nml, true_label)) if is_out_of_dist is False else nml_dict[
            'loss'].append(None)
        for prob_label, prob_single in enumerate(prob_nml):
            nml_dict[str(prob_label)].append(prob_single)
        nml_dict['log10_norm_factor'].append(np.log10(norm_factor))
        loc.append(int(keys))
        nml_dict['entropy'].append(entropy(prob_nml, base=10))

    # Create df
    pnml_df = pd.DataFrame(nml_dict, index=loc)

    # Add more columns
    is_correct = np.array(pnml_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        pnml_df['true_label']).astype(int)
    pnml_df['is_correct'] = is_correct
    return pnml_df


def result_dict_to_erm_df(results_dict, is_random_labels=False, is_out_of_dist=False):
    # Initialize columns to df
    cls_keys = list(filter(lambda l: l.isdigit(), list(results_dict['0'].keys())))  # extract only the digits keys
    cls_keys = [int(k) for k in cls_keys]
    df_col = [str(x) for x in range(min(cls_keys), max(cls_keys)+1)] + ['true_label', 'loss', 'entropy']
    erm_dict = {}
    for col in df_col:
        erm_dict[col] = []
    loc = []

    # Iterate on keys
    for keys in results_dict:
        # extract probability of test sample
        sample_dict = results_dict[keys]
        _, true_label, _, prob_org = extract_probabilities_list(sample_dict)
        true_label = true_label if is_random_labels is False else testloader.dataset.test_labels[int(keys)]
        erm_dict['true_label'].append(true_label)
        erm_dict['loss'].append(compute_log_loss(prob_org, true_label)) if is_out_of_dist is False else erm_dict[
            'loss'].append(None)
        for prob_label, prob_single in enumerate(prob_org):
            erm_dict[str(prob_label)].append(prob_single)
        erm_dict['entropy'].append(entropy(prob_org, base=10))
        loc.append(int(keys))

    # Create df
    erm_df = pd.DataFrame(erm_dict, index=loc)

    # Add more columns
    is_correct = np.array(erm_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        erm_df['true_label']).astype(int)
    erm_df['is_correct'] = is_correct
    return erm_df


def result_dict_to_genie_df(results_dict, is_random_labels=False):
    # Initialize columns to df
    cls_keys = list(filter(lambda l: l.isdigit(), list(results_dict['0'].keys())))  # extract only the digits keys
    cls_keys = [int(k) for k in cls_keys]
    df_col = [str(x) for x in range(min(cls_keys), max(cls_keys)+1)] + ['true_label', 'loss', 'entropy']
    genie_dict = {}
    for col in df_col:
        genie_dict[col] = []
    loc = []

    # Iterate on keys
    for keys in results_dict:
        # extract probability of test sample
        sample_dict = results_dict[keys]
        prob_genie, true_label, predicted_genie_label = extract_genie_probabilities_list(sample_dict)
        true_label = true_label if is_random_labels is False else testloader.dataset.test_labels[int(keys)]
        genie_dict['true_label'].append(true_label)
        genie_dict['loss'].append(compute_log_loss(prob_genie, true_label))
        for prob_label, prob_single in enumerate(prob_genie):
            genie_dict[str(prob_label)].append(prob_single)
        genie_dict['entropy'].append(entropy(prob_genie, base=10))
        loc.append(int(keys))

    # Create df
    genie_df = pd.DataFrame(genie_dict, index=loc)

    # Add more columns
    is_correct = np.array(genie_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        genie_df['true_label']).astype(int)
    genie_df['is_correct'] = is_correct

    return genie_df


def get_testset_intersections_from_results_dfs(results_df_list: list):
    if not results_df_list:
        print('Got empty list!')
        return [], None

    idx_common = set(results_df_list[0].index.values)
    for df_idx in range(1, len(results_df_list)):
        idx_common = idx_common & set(results_df_list[df_idx].index.values)

    for df_idx in range(len(results_df_list)):
        results_df_list[df_idx] = results_df_list[df_idx].loc[idx_common].sort_index()
    return results_df_list, idx_common


def create_twice_univ_df(results_df_list: list):
    if not results_df_list:
        print('Got empty list!')
        return [], None

    results_df_list, idx_common = get_testset_intersections_from_results_dfs(results_df_list)

    # Twice Dataframe creation . Take the maximum for each label
    twice_df = pd.DataFrame(columns=[str(x) for x in range(10)], index=idx_common)
    for label in range(10):

        # Extract label prob from each df
        prob_from_dfs = []
        for df in results_df_list:
            prob_from_dfs.append(df[str(label)])

        # Assign the maximum to twice df
        twice_df[str(label)] = np.asarray(prob_from_dfs).max(axis=0).tolist()

    # Normalize the prob
    normalization_factor = twice_df.sum(axis=1)
    twice_df = twice_df.divide(normalization_factor, axis='index')
    twice_df['log10_norm_factor'] = normalization_factor

    # Add true label column
    twice_df['true_label'] = results_df_list[0]['true_label'].astype(int)

    # assign is_correct columns
    is_correct = np.array(twice_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        twice_df['true_label'])
    twice_df['is_correct'] = is_correct

    # assign loss and entropy
    loss = []
    entropy_list = []
    for index, row in twice_df.iterrows():
        loss.append(-np.log10(row[str(int(row['true_label']))]))
        entropy_list.append(entropy(np.array(row[[str(x) for x in range(10)]]).astype(float)))
    twice_df['loss'] = loss
    twice_df['entropy'] = entropy_list

    return twice_df, idx_common

def create_genie_tu_df(results_df_list):
    """
    TU genie pick the lower loss between all models for each test point in the testset.
    """
    if not results_df_list:
        print('Got empty list!')
        return [], None

    results_df_list, idx_common = get_testset_intersections_from_results_dfs(results_df_list)

    # Twice Dataframe creation .
    genie_tu_df = pd.DataFrame.copy(results_df_list[0], deep=True)
    genie_tu_df = genie_tu_df.loc[idx_common]


    loss_list = []
    for df in results_df_list:
        loss_list.append(df['loss'])
    df_index_min_loss = np.argmin(loss_list, axis=0).tolist()  # find the dataframe index with the lowest loss

    for ii in idx_common:
        # print("choose df idx={}".format(df_index_min_loss[ii]))
        assert (np.isclose(results_df_list[df_index_min_loss[ii]].loc[ii][[str(x) for x in range(10)]].sum(), 1.0, rtol=1e-04))
        genie_tu_df.loc[ii] = results_df_list[df_index_min_loss[ii]].loc[ii]
    genie_tu_df["selected_df"] = df_index_min_loss

    return genie_tu_df, idx_common

def create_risk_minimizer_df(results_df_list):
    """ create_risk_minimizer_df - risk minimizer picks the lower risk between all models for each test point
    in the testset"""
    if not results_df_list:
        print('Got empty list!')
        return [], None

    results_df_list, idx_common = get_testset_intersections_from_results_dfs(results_df_list)

    # Twice Dataframe creation .
    risk_min_df = pd.DataFrame.copy(results_df_list[0], deep=True)
    risk_min_df = risk_min_df.loc[idx_common]


    risk_list = []
    for df in results_df_list:
        risk_list.append(df['log10_norm_factor'])
    df_index_min_risk = np.argmin(risk_list, axis=0).tolist()  # find the dataframe index with the lowest loss

    for ii in idx_common:
        # print("choose df idx={}".format(df_index_min_risk[ii]))
        risk_min_df.loc[ii] = results_df_list[df_index_min_risk[ii]].loc[ii]
    risk_min_df["selected_df"] = df_index_min_risk

    return risk_min_df, idx_common


def create_bagging_df(results_df_list):
    """create_bagging_df - Bagging is an ensemble learning technique in which the probabilities of all models are
    averaged."""
    if not results_df_list:
        print('Got empty list!')
        return [], None

    results_df_list, idx_common = get_testset_intersections_from_results_dfs(results_df_list)

    # Bagging dataframe creation. Average label predictions for each index (sample)
    bagging_df = pd.DataFrame(columns=[str(x) for x in range(10)], index=idx_common)
    bagging_df[[str(x) for x in range(10)]] = 0
    for i in range(len(results_df_list)):
        bagging_df = bagging_df.add(results_df_list[i][[str(x) for x in range(10)]])

    bagging_df = bagging_df.divide(len(results_df_list), axis='index')


    # Add true label column
    bagging_df['true_label'] = results_df_list[0]['true_label'].astype(int)

    # assign is_correct columns
    is_correct = np.array(bagging_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        bagging_df['true_label'])
    bagging_df['is_correct'] = is_correct

    # assign loss and entropy
    loss = []
    entropy_list = []
    for index, row in bagging_df.iterrows():
        loss.append(-np.log10(row[str(int(row['true_label']))]))
        entropy_list.append(entropy(np.array(row[[str(x) for x in range(10)]]).astype(float)))
        assert(np.isclose(row[[str(x) for x in range(10)]].sum(), 1.0, rtol=1e-04))
    bagging_df['loss'] = loss
    bagging_df['entropy'] = entropy_list

    return bagging_df, idx_common


def calc_erm_and_genie_stats(results_df_list):
    results_dict = load_dict_from_file_list(results_df_list)

    is_random_labels = False
    is_out_of_dist = False
    genie_df, statistic_genie_df = None, None

    # Calc genie
    genie_df = result_dict_to_genie_df(results_dict, is_random_labels=is_random_labels)
    statistic_genie_df = calc_statistic_from_df_single(genie_df).rename(columns={'statistics': 'genie'})
    genie_df = genie_df.add_prefix('genie_')
    print(statistic_genie_df)
    print(genie_df.shape[0])

    # Calc ERM
    erm_df = result_dict_to_erm_df(results_dict, is_random_labels=is_random_labels, is_out_of_dist=is_out_of_dist)
    statistic_erm_df = calc_statistic_from_df_single(erm_df).rename(columns={'statistics': 'erm'})
    erm_df = erm_df.add_prefix('erm_')
    print(statistic_erm_df)
    print(erm_df.shape[0])


def find_results_in_path(path_to_folder:str):
    pathname = path_to_folder + '/**/results*.json'
    return glob.glob(pathname,  recursive=True)





if __name__ == "__main__":
    # # Example
    # json_file_name = 'results_example.json'
    json_file_name = './../output/imagenet_adversarial_results_20190725_172246/results_imagenet_adversarial_20190725_172246.json'
    json_file_name = os.path.join('.', json_file_name)
    # with open(json_file_name) as data_file:
    #     results_dict_sample = json.load(data_file)
    # nml_df_sample = result_dict_to_nml_df(results_dict_sample)
    #
    # tic = time.time()
    result_df_sample, statistics_df_sample = load_results_to_df([json_file_name])
    print(statistics_df_sample.transpose())
    print("number of test samples:{}".format(result_df_sample.shape[0]))
    # print('load_results_to_df: {0:.2f} [s]'.format(time.time() - tic))
    # tic = time.time()
    # nml_df = result_dict_to_nml_df(results_dict_sample)
    # print('result_dict_to_nml_df: {0:.2f} [s]'.format(time.time() - tic))
    # tic = time.time()
    # statistic = calc_statistic_from_df_single(nml_df)
    # print('calc_statistic_from_df_single: {0:.2f} [s]'.format(time.time() - tic))
    #
    # a, b = create_twice_univ_df([nml_df, nml_df])
    # print('Done!')

    M11_path = ['./../results/deep_net/twice_univ/M11/results_mnist_adversarial_20190416_164500.json']
    M12_path = ['./../results/deep_net/twice_univ/M12/results_mnist_adversarial_20190416_164641.json']
    M13_path = ['./../results/deep_net/twice_univ/M13/results_mnist_adversarial_20190416_164806.json']
    M14_path = ['./../results/deep_net/twice_univ/M14A/results_mnist_adversarial_20190417_153920.json',
                './../results/deep_net/twice_univ/M14B/results_mnist_adversarial_20190417_153958.json']
    M21_path = ['./../results/deep_net/twice_univ/M21/results_mnist_adversarial_20190416_165007.json']
    M22_path = ['./../results/deep_net/twice_univ/M22/results_mnist_adversarial_20190416_165127.json']
    M23_path = ['./../results/deep_net/twice_univ/M23/results_mnist_adversarial_20190416_165303.json']
    M24_path = ['./../results/deep_net/twice_univ/M24A/results_mnist_adversarial_20190417_154137.json',
                './../results/deep_net/twice_univ/M24B/results_mnist_adversarial_20190417_154214.json']

    M1 = result_dict_to_nml_df(load_dict_from_file_list(M11_path))
    print('loaded 0 layers MNIST, {} samples'.format(M1.shape[0]))
    M2 = result_dict_to_nml_df(load_dict_from_file_list(M21_path))
    print('loaded 1 layers MNIST, {} samples'.format(M2.shape[0]))

    twice_df, idx_common = create_genie_tu_df([M1, M2])#create_twice_univ_df
    statistic_genie_df = calc_statistic_from_df_single(twice_df).rename(columns={'statistics': 'genie'})
    print(statistic_genie_df)