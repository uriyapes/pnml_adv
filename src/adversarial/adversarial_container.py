import torch
import os
from typing import Union, List


class AdversarialContainer(object):
    def __init__(self, attack_params: Union[dict, None], original_sample: Union[torch.Tensor, None], true_label: torch.Tensor, probability: torch.Tensor, loss: torch.Tensor,
                  adversarial_sample: Union[torch.Tensor, None] = None, genie_prob: Union[torch.Tensor, None] = None):
        """

        :param original_sample: The original data (non adversarial)
        :param true_label: The true label
        :param probability: The adversarial_sample (original_sample if adversarial_sample=None) probabilities for each label
        :param loss: cross-entropy loss (this is not calculated using probabilities because torch.log_softmax(logits))
                                        is more numerical stable than torch.log(probabilities))
        :param adversarial_sample: Adversarial sample (if exist)
        :param genie_prob: Genie probability from PnmlModel (if exist)
        """
        self.attack_params = attack_params
        # TODO: pre-allocate memory in advance using dataset shape
        self.original_sample = original_sample.detach().cpu() if original_sample is not None else None
        self.true_label = true_label.detach().cpu()
        self.adversarial_sample = adversarial_sample.detach().cpu() if adversarial_sample is not None else None
        self.probability = probability.detach().cpu()
        self.predict = torch.max(self.probability, 1)[1]
        self.correct = (self.predict == self.true_label)
        self.loss = loss.detach().cpu()
        if genie_prob is not None:
            self.genie_prob = genie_prob.detach().cpu()
            self.regret = torch.log(self.genie_prob.sum(dim=1, keepdim=False))
        else:
            self.genie_prob = None
            self.regret = None

    def merge(self, adv):
        self.original_sample = torch.cat([self.original_sample, adv.original_sample], dim=0) if adv.original_sample is not None else None
        self.true_label = torch.cat([self.true_label, adv.true_label])
        self.adversarial_sample = torch.cat([self.adversarial_sample, adv.adversarial_sample]) if adv.adversarial_sample is not None else None
        self.probability = torch.cat([self.probability, adv.probability])
        self.predict = torch.cat([self.predict, adv.predict])
        self.correct = torch.cat([self.correct, adv.correct])
        self.loss = torch.cat([self.loss, adv.loss])
        if self.genie_prob is not None:
            self.genie_prob = torch.cat([self.genie_prob, adv.genie_prob])
            self.regret = torch.cat([self.regret, torch.log(adv.genie_prob.sum(dim=1, keepdim=False))])

    @staticmethod
    def cat(adv_l: List):
        adv_l[0].original_sample = torch.cat([adv.original_sample for adv in adv_l], dim=0) if adv_l[0].original_sample is not None else None
        adv_l[0].true_label = torch.cat([adv.true_label for adv in adv_l])
        adv_l[0].adversarial_sample = torch.cat([adv.adversarial_sample for adv in adv_l]) if adv_l[0].adversarial_sample is not None else None
        adv_l[0].probability = torch.cat([adv.probability for adv in adv_l])
        adv_l[0].predict = torch.cat([adv.predict for adv in adv_l])
        adv_l[0].correct = torch.cat([adv.correct for adv in adv_l])
        adv_l[0].loss = torch.cat([adv.loss for adv in adv_l])
        if adv_l[0].genie_prob is not None:
            adv_l[0].genie_prob = torch.cat([adv.genie_prob for adv in adv_l])
            adv_l[0].regret = torch.cat([adv.regret for adv in adv_l])
        return adv_l[0]

    def dump(self, path_to_folder: str, file_name: str = "adversarials.t"):
        torch.save(self, os.path.join(path_to_folder, file_name))

    def get_mean_loss(self):
        return self.loss.sum().item() / len(self.loss)

    def get_accuracy(self):
        return float(self.correct.sum().item()) / len(self.correct)

