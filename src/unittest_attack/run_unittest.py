import json
import unittest
import os
import torch
from experimnet_utilities import Experiment
import logger_utilities
from eval import eval_adversarial_dataset
from utilities import TorchUtils


class TestModel(unittest.TestCase):
    def setUp(self):
        TorchUtils.set_rnd_seed(1)
        self.general_args = {'experiment_type': 'mnist_adversarial', 'output_root': './src/unittest_attack/logs',
                             'param_file_path': None}
        self.test_name = self.id().split('.')[-1]
        logger_utilities.init_logger(logger_name=self.test_name, output_root=self.general_args['output_root'])
        self.logger = logger_utilities.get_logger()

    def tearDown(self):
        self.logger.info("%s Done" % (self.test_name))
        logger_utilities.delete_logger()

    def _replicate_results(self):
        self.logger.info("Start: {} expected_result_folder: {}".format(self.test_name, self.expected_result_folder))
        adv_expected = torch.load(os.path.join(self.expected_result_folder, 'adversarials.t'))

        model_to_eval, dataloaders, attack = self._prepare_test()
        adv = eval_adversarial_dataset(model_to_eval, dataloaders['test'], attack)

        self.logger.info("Accuracy: {}, Loss: {}".format(adv.get_accuracy(), adv.get_mean_loss()))
        # self.assertEqual(adv.attack_params, adv_expected.attack_params)
        self.assertTrue(torch.equal(adv.predict, adv_expected.predict))
        self.assertTrue(torch.equal(adv.correct, adv_expected.correct))
        self.assertTrue(torch.equal(adv.loss, adv_expected.loss))
        if adv.genie_prob is not None:
            self.assertTrue(torch.equal(adv.genie_prob, adv_expected.genie_prob))
            self.assertTrue(torch.equal(adv.regret, adv_expected.regret))

    def _prepare_test(self):
        self.general_args['param_file_path'] = os.path.join(self.expected_result_folder, 'params.json')
        exp = Experiment(self.general_args)
        with open(os.path.join(self.logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
            outfile.write(json.dumps(exp.params, indent=4, sort_keys=False))
        self.logger.info(exp.params)

        model_to_eval = exp.get_model(exp.params['model']['model_arch'], exp.params['model']['ckpt_path'],
                                      exp.params['model']['pnml_active'])
        data_folder = "./data"
        dataloaders = exp.get_adv_dataloaders(data_folder)
        # Get adversarial attack:
        attack = exp.get_attack_for_model(model_to_eval)
        return model_to_eval, dataloaders, attack

    def test_mnist_pgd_no_pnml(self):
        self.expected_result_folder = os.path.join('src/unittest_attack', 'test_mnist_pgd_no_pnml_expected_result')
        self._replicate_results()

    def test_mnist_pgd_with_pnml(self):
        self.expected_result_folder = os.path.join('src/unittest_attack', 'test_mnist_pgd_with_pnml_expected_result')
        self._replicate_results()


if __name__ == '__main__':
    unittest.main()
