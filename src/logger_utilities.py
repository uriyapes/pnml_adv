import json
import logging
import os
import sys
import time
import pickle
import pathlib


class Logger:
    def __init__(self, experiment_type: str, output_root: str):
        """
        Initialize logger class
        :param experiment_type: the experiment type- use for saving string of the outputs/
        :param output_root: the directory to which the output will be saved.
        """

        # Create logger
        logger = logging.getLogger("utilities_logger")
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.logger = logger
        self.json_file_name = None
        self.results_dict = {}

        self.unique_time = time.strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(output_root, '%s_results_%s' %
                                          (experiment_type, self.unique_time))
        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.define_log_file(os.path.join(self.output_folder, 'log_%s_%s.log' %
                                          (experiment_type, self.unique_time)))
        self.define_json_output(os.path.join(self.output_folder, 'results_%s_%s.json' %
                                             (experiment_type, self.unique_time)))
        self.time = None

    def define_log_file(self, log_file_name: str):
        """
        create log file to be save into hard disk
        :param log_file_name: the name of the log file
        :return:
        """
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def info(self, string_to_print: str):
        """
        print and save to log file in logger style info
        :param string_to_print: string that will be display in the log
        :return:
        """
        self.logger.info(string_to_print)

    def debug(self, string_to_print: str, measure_time: bool = True):
        """
        print and save to log file in logger style info with DEBUG priority
        :param string_to_print: string that will be display in the log
        :return:
        """
        if self.time is not None and measure_time is True:
            new_time = time.time()
            string_to_print = string_to_print + ", time: {}".format(new_time - self.time)
            self.time = new_time
        self.logger.debug(string_to_print)

    def init_debug_time_measure(self):
        self.time = time.time()

    def define_json_output(self, json_file_name: str):
        """
        set the output json file name. The results of the PNML will be save into.
        :param json_file_name: the file name of the results file
        :return:
        """
        self.json_file_name = json_file_name

    def save_json_file(self):
        """
        Save results into hard disk
        :return:
        """
        with open(self.json_file_name, 'w') as outfile:
            json.dump(self.results_dict,
                      outfile,
                      sort_keys=True,
                      indent=4,
                      ensure_ascii=False)

    def add_entry_to_results_dict(self, test_idx_sample, prob_key_str, prob,
                                  train_loss, test_loss):
        """
        Add results entry into the result dict
        :param test_idx_sample: the test sample index in the testset.
        :param prob_key_str: the label of the test sample that the model was trained with.
        :param prob: the predicted probability assignment.
        :param train_loss: the loss of the trainset after training the model with the test sample
        :param test_loss: the loss of the testset after training the model with the test sample
        :return:
        """
        if str(test_idx_sample) not in self.results_dict:
            self.results_dict[str(test_idx_sample)] = {}

        self.results_dict[str(test_idx_sample)][prob_key_str] = {}
        self.results_dict[str(test_idx_sample)][prob_key_str]['prob'] = prob
        self.results_dict[str(test_idx_sample)][prob_key_str]['train_loss'] = train_loss
        self.results_dict[str(test_idx_sample)][prob_key_str]['test_loss'] = test_loss

    def add_org_prob_to_results_dict(self, test_idx_sample, prob_org, true_label):
        """
        Adding the ERM base model probability assignment on the test sample.
        :param test_idx_sample: the test sample index in the testset.
        :param prob_org: the predicted probability assignment
        :param true_label: the true label of the test sample
        :return:
        """
        if str(test_idx_sample) not in self.results_dict:
            self.results_dict[str(test_idx_sample)] = {}

        self.results_dict[str(test_idx_sample)]['original'] = {}
        self.results_dict[str(test_idx_sample)]['original']['prob'] = prob_org
        self.results_dict[str(test_idx_sample)]['true_label'] = int(true_label)

    def dump_pickle(self, obj):
        file_path = os.path.join(self.output_folder, 'adversarials.p')
        pickle.dump(obj, open(file_path, "wb"))

    @classmethod
    def load_pickle(cls, path):
        file_path = os.path.join(path, 'adversarials.p')
        return pickle.load(open(file_path, "rb"))