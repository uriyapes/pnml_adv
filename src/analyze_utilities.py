import json
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import entropy
import glob
pd.set_option('display.max_columns', 500)
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
    return -np.log(normalized_prob[true_label] + np.finfo(float).tiny)


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


def extract_results_from_logfile_to_df(log_file):
    """
    Get a log file, using regular expression read the evaluation (accuracy and loss of the test set) and return them as
    a dataframe
    :param log_file: path to log file
    :return: dataframe containing the results
    """
    txt_to_search = "Base model ----- \[Natural-train test\] loss .*"
    with open(log_file) as f:
        data = f.read()
        line = re.findall(txt_to_search,data)
        # print("Line is" + line[0] +"\nLines found:" + str(len(line)))
        assert(len(line) == 1)
        r = re.findall("[-+]?\d*\.\d+|\d+", line[0])
    statistics_df = pd.DataFrame(
        {'nml': pd.Series([float(r[3]), float(r[1])],
                                 index=['acc', 'mean loss'])})
    return statistics_df



def load_results_to_df(files, is_random_labels=False, is_out_of_dist=False, idx=None, flag_concat_results_df=True):
    results_dict = load_dict_from_file_list(files)

    # NML
    pnml_df = result_dict_to_nml_df(results_dict, is_random_labels=is_random_labels, is_out_of_dist=is_out_of_dist)
    if idx is None: # if not idx is given (empty set) then take all the indexes
        idx = pnml_df.index.values
    else:
        idx = set(idx)
    pnml_df = pnml_df.loc[idx]
    statistic_pnml_df = calc_statistic_from_df_single(pnml_df.loc[idx]).rename(columns={'statistics': 'nml'})
    # pnml_df = pnml_df.rename(columns={'nml_log_norm_factor': 'log_norm_factor'}) #TODO: remove this line

    # ERM
    erm_df = result_dict_to_erm_df(results_dict, is_random_labels=is_random_labels, is_out_of_dist=is_out_of_dist)
    erm_df = erm_df.loc[idx]
    statistic_erm_df = calc_statistic_from_df_single(erm_df.loc[idx]).rename(columns={'statistics': 'erm'})

    # genie
    genie_df, statistic_genie_df = None, None
    if is_out_of_dist is False:
        genie_df = result_dict_to_genie_df(results_dict, is_random_labels=is_random_labels)
        genie_df = genie_df.loc[idx]
        statistic_genie_df = calc_statistic_from_df_single(genie_df.loc[idx]).rename(columns={'statistics': 'genie'})

    # Merge and return
    result_df = pd.concat([pnml_df.add_prefix('nml_'), erm_df.add_prefix('erm_'), genie_df.add_prefix('genie_')], axis=1)
    statistic_df = pd.concat([statistic_pnml_df, statistic_erm_df, statistic_genie_df], axis=1)
    if flag_concat_results_df:
        return result_df, statistic_df
    else:
        return pnml_df, erm_df, genie_df, statistic_df


def comb_erm_nml_df_by_risk(erm_df, nml_df, risk_th):
    """
    comb_erm_nml_results_by_risk - takes two DF and combine them , element-wise, according to the risk. If the risk of a
    certain sample is smaller than risk_th then nml probabilities are used, else the normal probabilities (erm) are used.
    :param erm_df: DF containing the normal model probabilities.
    :param nml_df: DF containing the probabilities of pNML
    :param risk_th: The risk threshold
    :return: The combined Dataframe
    """
    assert(pd.Series(nml_df.index.values == erm_df.index.values).all())  # Make sure index are aligned.

    # Make sure both DF have the same column names
    # comb_df = nml_df.rename(columns=lambda x: x.replace("nml", "comb"))
    # erm_df = erm_df.rename(columns=lambda x: x.replace("erm", "comb"))
    comb_df = nml_df.copy()
    col = list(erm_df.columns)
    use_erm_idx = nml_df.log_norm_factor > risk_th
    comb_df.loc[use_erm_idx, col] = erm_df.loc[use_erm_idx, col]
    return comb_df


def find_optimal_risk_th_for_comb_df(files, min_risk_th:float = -0.1, max_risk_th:float = 1.1, risk_th_steps:int=30, idx=None):
    """
    find_optimal_risk_th_for_comb_df - create multiple number of combinations between erm and nml df according to the risk.
    :param files: the result files that contains the model accuracy
    :param risk_th_steps: the number of risk thresholds to test
    :param idx: the indexs to analyze in the dataframe, if no value is given all indexs are analyzed.
    :return:
    """
    results_dict = load_dict_from_file_list(files)

    # NML
    pnml_df = result_dict_to_nml_df(results_dict)
    if idx is None: # if not idx is given (empty set) then take all the indexes
        idx = pnml_df.index.values
    else:
        idx = set(idx)
    statistic_pnml_df = calc_statistic_from_df_single(pnml_df.loc[idx]).rename(columns={'statistics': 'nml'})
    # pnml_df = pnml_df.add_prefix('nml_')
    # pnml_df = pnml_df.rename(columns={'nml_log_norm_factor': 'log_norm_factor'})

    # ERM
    erm_df = result_dict_to_erm_df(results_dict)
    statistic_erm_df = calc_statistic_from_df_single(erm_df.loc[idx]).rename(columns={'statistics': 'erm'})
    # erm_df = erm_df.add_prefix('erm_')
    # max_risk, min_risk = max(pnml_df.log_norm_factor), min(pnml_df.log_norm_factor)
    risk_th_arr = np.arange(min_risk_th, max_risk_th, float(max_risk_th - min_risk_th)/risk_th_steps)
    statistics_df = pd.DataFrame(columns=['acc', 'mean loss', 'std loss', 'mean entropy', 'risk_th'])
    for iter, risk in enumerate(risk_th_arr):
        comb_df = comb_erm_nml_df_by_risk(erm_df, pnml_df, risk)
        statistic_line_name = 'comb_{}'.format(iter)
        statistic_comb_df = calc_statistic_from_df_single(comb_df.loc[idx]).rename(columns={'statistics': statistic_line_name})
        statistic_comb_df = statistic_comb_df.transpose()
        statistic_comb_df['risk_th'] = risk
        statistics_df = pd.concat([statistics_df, statistic_comb_df], ignore_index=False, sort=False)
    # statistics_df = pd.concat([statistics_df, statistic_erm_df.transpose(), statistic_pnml_df.transpose()], ignore_index=False, sort=False)
    max_acc = max(statistics_df.acc)

    print("maximum accuracy is: {}\n {}".format(max_acc, statistics_df.loc[statistics_df.acc == max_acc]))
    return statistics_df


def create_adv_detector_df(result_df, threshold, detect_method="risk", adv_dataset_flag=True, idx=None):
    """
    create_adv_detector_df - gets result_df and by using the risk detects misclassified samples, the misclassification
    could be a result of wrong classification of natural or adversarial image. In case the detected sample is truly wrong
    (TN - true negative) the sample is saved as correctly classified. In case the detected sample was classified correctly
    then the detector reclassifiy it as incorrect.
    The detection is performed by examining the risk of each sample, if the risk is higher than the threshold then the sample
    is considered wrongly classified.
    :param result_df: a dataframe containing the results
    :param threshold: the risk threshold/ probability threshold, samples with higher risk / lower probability  than threshold are detected as wrongly classified.
    :param detect_method: If "risk" use the risk to detect adversarial samples. If "prob" use max-probability for detection
    :param idx: the indexs to analyze in the dataframe, if no value is given all indexs are analyzed.
    :param adv_dataset_flag: indicates whether the dataset is adversarial or natural
    :return: A statistics dataframe after the reclassification according to the risk
    """
    assert(detect_method == "risk" or detect_method == "prob")
    # print(result_df)
    if idx is None: # if not idx is given (empty set) then take all the indexes
        idx = result_df.index.values
    else:
        idx = set(idx)

    result_df = pd.DataFrame.copy(result_df.loc[idx], deep=True)
    if detect_method == "risk":
        detected_idx = result_df['log_norm_factor'] > threshold
    elif detect_method == "prob":
        cls_keys = list(filter(lambda l: l.isdigit(), [str(col) for col in result_df.keys()]))
        detected_idx = result_df[cls_keys].apply(lambda row: row.max() < threshold, axis=1)
    # print(detected_idx)
    positive_detcted_idx = (detected_idx) & (result_df['is_correct'] == False)
    if adv_dataset_flag:
        TP = len(result_df.loc[detected_idx])
        FN = len(idx) - TP
        TPSA = len(result_df.loc[positive_detcted_idx])
        FP = 0
        # print(TP,FN, TPSA)
        result_df.loc[detected_idx, "is_correct"] = True
    else:
        correct_fp_idx = (detected_idx) & (result_df['is_correct'] == True)
        CFP = len(result_df.loc[correct_fp_idx])  # correct false-positive , measures how many correct samples are detected
        FP = len(result_df.loc[detected_idx])
        TN = len(idx) - FP
        # print(TN, FP)
        result_df.loc[detected_idx, "is_correct"] = False
        # result_df.loc[detected_idx, "is_correct"] = ~result_df.loc[detected_idx, "is_correct"] # ~ is element-wise NOT operator, correct become incorrect in vice versa

    statistic_adv_detector_df = calc_statistic_from_df_single(result_df.loc[idx]).rename(columns={'statistics': 'acc'})
    statistic_adv_detector_df.loc["threshold"] = threshold
    statistic_adv_detector_df.loc["FPR"] = float(FP) / len(idx)

    return statistic_adv_detector_df


def calc_statistic_from_df_single(result_df):
    mean_loss, std_loss = result_df['loss'].mean(), result_df['loss'].std()
    acc = result_df['is_correct'].sum() / result_df.shape[0]
    mean_entropy = np.mean(result_df['entropy'])
    statistics_df = pd.DataFrame(
        {'statistics': pd.Series([acc, mean_loss, std_loss, mean_entropy],
                                 index=['acc', 'mean loss', 'std loss', 'mean entropy'])})
    return statistics_df


def calc_nml_change_per_label(nml_df, erm_df, log_flag = False, unnormallized_nml_flag = True):
    """
    calc_nml_change_per_label - calculates the difference between the probabilities/log-loss of nml and erm predictions
                                for all classes.
    :param nml_df: pNML dataframe that contains the probabilities of the different labels.
    :param erm_df: ERM dataframe that contains the probabilities of the different labels.
    :return diff_df: A dataframe that contains the difference in prob./log-loss of each sample for each one of the
                    possible labels.
    """
    assert(len(nml_df) == len(erm_df))
    assert(~log_flag or ~unnormallized_nml_flag)  # No reason to calculate unnormalized mean loss
    # df_cols = [str(x) for x in range(10)]
    # df_col = [str(x) for x in ]


    cls_keys = list(filter(lambda l: l.isdigit(), [str(col) for col in nml_df.keys()] ))
    diff_df = pd.DataFrame(columns=cls_keys)
    diff_df['true_label'] = nml_df['true_label']
    diff_df['is_correct_nml'] = nml_df['is_correct']
    diff_df['is_correct_erm'] = erm_df['is_correct']
    second_largest_idx = erm_df[cls_keys].apply(lambda row: row.nlargest(2).idxmin(), axis=1)  # Find the second largest idx
    diff_df['other_label'] = erm_df[cls_keys].apply(lambda row: row.nlargest(3).idxmin(), axis=1)  # Find the third largest idx
    if unnormallized_nml_flag:
        nml_df.loc[:, cls_keys] = nml_df.apply(lambda row: row[cls_keys] * np.exp(row['log_norm_factor']), axis=1)[cls_keys]
    if log_flag:
        nml_df.loc[:,cls_keys] = -np.log(nml_df[cls_keys] + np.finfo(float).tiny)
        erm_df.loc[:,cls_keys] = -np.log(erm_df[cls_keys] + np.finfo(float).tiny)
    diff_df.loc[:,cls_keys] = nml_df[cls_keys] - erm_df[cls_keys]
    # Find adversarial label - the highest probability label (which isn't the true label)
    diff_df['adv_label'] = erm_df[[str(x) for x in range(10)]].idxmax(axis=1)
    idxs_to_replace_to_second_largest = diff_df['adv_label'].astype('int32') == diff_df['true_label'].astype('int32')
    print("number of idxs to replace to second largest: {}".format(idxs_to_replace_to_second_largest.sum()))

    diff_df.loc[idxs_to_replace_to_second_largest, 'adv_label'] = second_largest_idx
    diff_df['true_minus_adv_improve'] = diff_df.apply(lambda row: row[str(row['true_label'])] - row[str(row['adv_label'])], axis=1)
    diff_df['true_label_diff'] = diff_df.apply(lambda row: row[str(row['true_label'])], axis=1)
    diff_df['adv_label_diff'] = diff_df.apply(lambda row: row[str(row['adv_label'])], axis=1)
    diff_df['other_label_diff'] = diff_df.apply(lambda row: row[str(row['other_label'])], axis=1)
    return diff_df

def result_dict_to_nml_df(results_dict, is_random_labels=False, is_out_of_dist=False):
    # Initialize col of df
    first_idx = next(iter(results_dict.keys()))
    cls_keys = list(filter(lambda l: l.isdigit(), list(results_dict[first_idx].keys())))  # extract only the digits keys
    cls_keys = [int(k) for k in cls_keys]
    df_col = [str(x) for x in range(min(cls_keys), max(cls_keys)+1)] + \
             ['true_label', 'loss', 'log_norm_factor', 'entropy']
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
        nml_dict['log_norm_factor'].append(np.log(norm_factor))
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
    first_idx = next(iter(results_dict.keys()))
    cls_keys = list(filter(lambda l: l.isdigit(), list(results_dict[first_idx].keys())))  # extract only the digits keys
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
    first_idx = next(iter(results_dict.keys()))
    cls_keys = list(filter(lambda l: l.isdigit(), list(results_dict[first_idx].keys())))  # extract only the digits keys
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
    twice_df['log_norm_factor'] = normalization_factor

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
        loss.append(-np.log(row[str(int(row['true_label']))]))
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
        risk_list.append(df['log_norm_factor'])
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
        loss.append(-np.log(row[str(int(row['true_label']))]))
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


def create_nml_vs_eps_df(path_to_root_dir:str, eps_type:str = 'fix', idx = None, flag_read_json=True):
    """
    Plot a graph of Accuracy as a function of epsilon.
    :param path_to_root_dir: Path to directory that contains all the subdirectories with the results
    :param eps_type: Determine the X-axis, 'fix' means to take the eps of the refinement and 'attack' means to take the
                     eps of the attack.
    """
    search_string = path_to_root_dir + '\\*'
    subdir_list = glob.glob(search_string, recursive=False)
    assert(len(subdir_list) >= 1)
    for iter, dir in enumerate(subdir_list):
        statistics_df = load_results_to_df_with_params(dir, idx=idx, eps_type=eps_type, flag_read_json=flag_read_json)
        if statistics_df is None:
            continue
        if iter == 0:
            results_summary_df = pd.DataFrame(columns=statistics_df.index.tolist())
            results_summary_erm_df = pd.DataFrame(columns=statistics_df.index.tolist())
        results_summary_df = pd.concat([results_summary_df, statistics_df[['nml']].transpose()],
                                       ignore_index=True, sort=False)
        if flag_read_json:
            results_summary_erm_df = pd.concat([results_summary_erm_df, statistics_df[['erm']].transpose()],
                                               ignore_index=True, sort=False)
    return results_summary_df.sort_values('eps'), results_summary_erm_df.sort_values('eps')


def load_results_to_df_with_params(dir, idx=None, eps_type:str = 'fix', flag_return_res_df=False, flag_read_json=True):
    """
    Load result to dataframe with experiment parameters.
    :param dir: The parent directory
    :param idx: which idxs to compute from .json file
    :param eps_type: 'fix' for refinement eps (\lambda in paper) or 'attack' for adversarial eps (\eps in paper)
    :param flag_return_res_df: see return
    :param flag_read_json: extract results from .json or from .log file
    :return: statistics_df (and pnml_df, erm_df, genie_df if flag_return_res_df is True)
    """
    assert(not (flag_read_json is False and flag_return_res_df is True))
    assert(not (flag_read_json is False and idx is not None))
    if type(dir) == list:
        assert(len(dir) == 1)
        dir = dir[0]
    else:
        assert(type(dir) == str)
    if flag_read_json:
        results_path = dir + '\\results**.json'
    else:
        results_path = dir + '\\log**.log'
    results_path = glob.glob(results_path, recursive=True)
    assert (len(results_path) < 2)
    if len(results_path) == 0:
        print("Warning: directory: " + dir + "is empty")
        return None
    print("Loading:" + str(results_path))
    if flag_read_json:
        pnml_df, erm_df, genie_df, statistics_df = load_results_to_df(results_path, idx=idx, flag_concat_results_df=False)
    else:
        statistics_df = extract_results_from_logfile_to_df(results_path[0])

    param_path = dir + '\\params.json'
    param_path = glob.glob(param_path, recursive=True)
    assert (len(param_path) == 1)
    with open(param_path[0]) as f:
        params = json.load(f)
    statistics_df = statistics_df.transpose()

    statistics_df.loc['nml', 'eps'] = params['fit_to_sample']['epsilon'] if eps_type == 'fix' else \
        params['adv_attack_test']['epsilon']
    statistics_df.loc['erm', 'eps'] = params['fit_to_sample']['epsilon'] if eps_type == 'fix' else \
        params['adv_attack_test']['epsilon']

    statistics_df.loc['nml', 'refine_iter'] = params['fit_to_sample']['pgd_iter']
    statistics_df.loc['nml', 'refine_random_start'] = params['fit_to_sample']['pgd_rand_start']
    statistics_df.loc['nml', 'refine_restart_num'] = params['fit_to_sample']['pgd_test_restart_num']
    statistics_df.loc['nml', 'beta'] = params['adv_attack_test']['beta'] if params['adv_attack_test'].keys().__contains__('beta') else 0

    statistics_df.loc['nml', 'params'] = [params]  # To insert dict to DF we need to put it inside list
    statistics_df.loc['erm', 'params'] = [params]

    statistics_df.loc['nml', 'params_fix_hash'] = hash(json.dumps(params['fit_to_sample'], sort_keys=True))
    statistics_df.loc['erm', 'params_fix_hash'] = hash(json.dumps(params['fit_to_sample'], sort_keys=True))

    statistics_df.loc['nml', 'results_path'] = results_path
    statistics_df.loc['erm', 'results_path'] = results_path  # To insert dict to DF we need to put it inside list
    if flag_return_res_df:
        pnml_df.columns = [col.replace('nml_', '') for col in pnml_df.columns]
        return pnml_df, erm_df, genie_df, statistics_df.transpose()
    else:
        return statistics_df.transpose()


def find_path_to_result_with_similar_params(path_to_result_dir, look_at_path, fix_params_comparison=['epsilon'], return_res_file_flag=True):
    """
    find_path_to_result_with_similar_params - get a path to a result directory, extracts the parameters used in this experiment,
    and look at look_at_path for a similiar param file according to fix_params_comparison list of parameters to compare.
    :param path_to_result_dir: the directory containing a param file. The the function looks for similiar param files to that one.
    :param look_at_path: Where to look for similiar param files.
    :param fix_params_comparison: A list of strings representing the param keys that are used for comparison
    :return: a list of paths to results.json which are inside directories containing the a similiar param file, None if nothing is found
    """
    results_path_l = []
    param_path = path_to_result_dir + '\\params.json'
    param_path = glob.glob(param_path, recursive=False)
    assert(len(param_path) == 1)
    with open(param_path[0]) as f:
        params = json.load(f)

    search_string = look_at_path + '\\*'
    subdir_list = glob.glob(search_string, recursive=False)
    for iter, dir in enumerate(subdir_list):
        # Make sure the dir isn't empty
        results_path = dir + '\\results**.json'
        results_path = glob.glob(results_path, recursive=False)
        assert(len(results_path) < 2)
        if len(results_path) == 0:
            print("Warning: look at path directory: " + dir + " is empty")
            continue

        # Extract param file
        param_path = dir + '\\params.json'
        param_path = glob.glob(param_path, recursive=False)
        assert(len(param_path) == 1)
        with open(param_path[0]) as f:
            params_to_compare = json.load(f)

        # Compare params according to key list fix_params_comparison
        identical_flag = True
        for key in fix_params_comparison:
            if params['fit_to_sample'][key] != params_to_compare['fit_to_sample'][key]:
                identical_flag = False
        #
        if identical_flag:
            if return_res_file_flag:
                results_path_l.append(results_path[0])
            else:
                results_path_l.append(dir)
        #     return results_path
        # else:
        #     results_path = None

    if len(results_path_l) == 0:
        print("No match in directory: {} for eps1: {}".format(dir, params['fit_to_sample'][key]))
        results_path_l = None
    return results_path_l


def create_list_of_corresponding_results_by_params(path1, path2, fix_params_comparison=['epsilon'], paths_to_res_file_flag=True):
    """

    :param path1:
    :param path2:
    :param fix_params_comparison:
    :return: results_l a list of results in path1 and corresponding_result_path_lol, a list of lists of corresponding results in path2
    """
    search_string = path1 + '\\*'
    subdir_list = glob.glob(search_string, recursive=False)
    # print(subdir_list)
    results_l = []
    corresponding_result_path_lol = []
    for iter, dir in enumerate(subdir_list):
        # Make sure directory isn't empty
        # print("iteration {}".format(iter))
        results_path = dir + '\\results**.json'
        results_path = glob.glob(results_path, recursive=True)
        assert (len(results_path) < 2)
        if len(results_path) == 0:
            print("Warning: directory: " + dir + "is empty")
            continue

        corresponding_result_path_l = find_path_to_result_with_similar_params(dir, path2, fix_params_comparison, paths_to_res_file_flag)
        if corresponding_result_path_l is not None:
            path_to_add = results_path[0] if paths_to_res_file_flag else dir
            results_l.append(path_to_add)
            corresponding_result_path_lol.append(corresponding_result_path_l)
        else:
            print("Warning: directory: " + dir + "didn't find a corresponding directory")
    return results_l, corresponding_result_path_lol





if __name__ == "__main__":
    json_file_name = './../results/paper/MNIST/mnist_adversarial_results_20190802_151544/results_mnist_adversarial_20190802_151544.json'
    # comb_df_statistics = find_optimal_risk_th_for_comb_df([json_file_name])
    # print(comb_df_statistics)
    results_dict = load_dict_from_file_list([json_file_name])
    nml_df = result_dict_to_nml_df(results_dict)
    erm_df = result_dict_to_erm_df(results_dict)
    # diff_df = calc_nml_change_per_label(nml_df, erm_df)
    # print(diff_df)
    statistics_detector = create_adv_detector_df(erm_df, 0, detect_method="prob")

    results_summary_df = create_nml_vs_eps_df('./../results/cifar/acc_vs_eps_refine/cifar_diff_fix')

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