# Code is based on https://github.com/oscarknagg/adversarial
from abc import ABC, abstractmethod
from adversarial.functional import *
import torch
import os
from typing import Union


class Adversarials(object):
    def __init__(self, attack_params: dict, original_sample: Union[torch.Tensor, None], true_label: torch.Tensor, probability: torch.Tensor, loss: torch.Tensor,
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

    def dump(self, path_to_folder):
        torch.save(self, os.path.join(path_to_folder, "adversarials.t"))

    def get_mean_loss(self):
        return self.loss.sum() / len(self.loss)

    def get_accuracy(self):
        return self.correct.sum().item() / len(self.correct)


class Attack(ABC):
    """Base class for adversarial attack methods"""
    @abstractmethod
    def create_adversarial_sample(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class NoAttack(Attack):
    def __init__(self):
        super(NoAttack, self).__init__()
        self.name = self.__class__.__name__

    def create_adversarial_sample(self, x: torch.Tensor, y: torch.Tensor,  y_target: torch.Tensor = None):
        return x


class Natural(Attack):
    def __init__(self, model):
        super(Natural, self).__init__()
        self.name = self.__class__.__name__
        self.model = model

    def create_adversarial_sample(self, x: torch.Tensor, y: torch.Tensor, y_target: torch.Tensor = None,
                                  get_adversarial_class: bool = False):
        loss, prob, genie_prob = self.model.eval_batch(x, y, self.model.pnml_model)
        if get_adversarial_class:
            return Adversarials(None, None, y, prob, loss, x, genie_prob)
        else:
            return x


class FGSM(Attack):
    """Implements the Fast Gradient-Sign Method (FGSM).

    FGSM is a white box attack.

    """
    def __init__(self,
                 model: Module,
                 loss_fn: Callable,
                 eps_ratio: float,
                 clamp: Tuple[float, float] = (0, 1)):
        """
        :param model: the model which will be used to create the adversarial examples, pay attention that as the model
        changes (by training) so is the adversarial examples created
        :param loss_fn: the loss function which the adversarial example tries to maximize
        :param eps_ratio: the L_infinity norm max value units of the total range of the allowed change (meaning that the
        epsilon value is adjusted for different images value ranges).
        :param: clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST
        """
        super(FGSM, self).__init__()
        self.name = self.__class__.__name__
        self.model = model
        self.loss_fn = loss_fn()
        self.eps_ratio = eps_ratio
        self.clamp = clamp
        self.eps = self.eps_ratio * (self.clamp[1] - self.clamp[0])

    def create_adversarial_sample(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  y_target: torch.Tensor = None
                                  ):
        """Creates an adversarial sample

        Args:
            :param: x: Batch of samples
            :param: y: Corresponding labels

        Returns:
            x_adv: Adversarially perturbed version of x
        """
        return fgsm(self.model, x, y, self.loss_fn, self.eps, y_target, self.clamp)


class PGD(Attack):
    """Implements the iterated Fast Gradient-Sign Method"""
    def __init__(self,
                 model: Module,
                 loss_fn: Callable,
                 attack_params: dict,
                 clamp: Tuple[float, float] = (0, 1),
                 norm: Union[str, int] = 'inf',
                 flip_grad_ratio=0.0):
        super(PGD, self).__init__()
        self.name = self.__class__.__name__
        self.model = model
        self.loss_fn = loss_fn  # TODO: not used currently
        self.attack_params = attack_params.copy()
        if attack_params["pgd_test_restart_num"] == 0:
            self.attack_params["random"] = False
            self.attack_params["pgd_test_restart_num"] = 1
        else:
            self.attack_params["random"] = True
        self.attack_params["epsilon"] = self.attack_params["epsilon"] * (clamp[1] - clamp[0])
        self.attack_params["pgd_step"] = self.attack_params["pgd_step"] * (clamp[1] - clamp[0])

        self.attack_params["norm"] = norm
        self.attack_params["clamp"] = clamp

        self.attack_params["flip_grad_ratio"] = flip_grad_ratio

    def create_adversarial_sample(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  y_target: torch.Tensor = None,
                                  get_adversarial_class: bool = False) -> Union[torch.Tensor, Adversarials]:
        adv_sample, adv_loss, adv_pred, genie_pred = iterated_fgsm(
                self.model, x, y, self.loss_fn, self.attack_params["pgd_iter"],self.attack_params["pgd_step"],
                self.attack_params["epsilon"], self.attack_params["norm"], y_target=y_target, random=self.attack_params["random"],
                clamp=self.attack_params["clamp"], restart_num=self.attack_params["pgd_test_restart_num"],
                beta=self.attack_params["beta"], flip_grad_ratio=self.attack_params["flip_grad_ratio"])
        if get_adversarial_class:
            adversarials = Adversarials(self.attack_params, x, y, adv_pred, adv_loss, adv_sample, genie_pred)
            return adversarials
        else:
            return adv_sample


class Boundary(Attack):
    """Implements the boundary attack

    This is a black box attack that doesn't require knowledge of the model
    structure. It only requires knowledge of

    https://arxiv.org/pdf/1712.04248.pdf

    Args:
        model:
        k:
        orthogonal_step: orthogonal step size (delta in paper)
        perpendicular_step: perpendicular step size (epsilon in paper)
    """
    def __init__(self, model: Module, k: int, orthogonal_step: float = 0.1, perpendicular_step: float = 0.1):
        super(Boundary, self).__init__()
        self.model = model
        self.k = k
        self.orthogonal_step = orthogonal_step
        self.perpendicular_step = perpendicular_step

    def create_adversarial_sample(self,
                                  model: Module,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  initial: torch.Tensor = None,
                                  clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        return boundary(model, x, y, self.orthogonal_step, self.perpendicular_step, self.k, initial, clamp)


def get_attack(attack_params: dict, model: Module = None, clamp: Tuple[float, float] = (0, 1),
               loss_fn: Callable = torch.nn.CrossEntropyLoss, flip_grad_ratio=0.0):
    """
    :param attack_params: a dictionary containing attack parameters. Dictionary keys:
        attack_type
        epsilon
        pgd_iter
        pgd_step
        pgd_test_restart_num
        beta
    :param model: The model to be attacked
    :param clamp: The value range of the samples
    :param  loss_fn:
    :param flip_grad_ratio:
    """
    if attack_params["attack_type"] == 'no_attack':
        attack = NoAttack()
    elif attack_params["attack_type"] == "natural":
        attack = Natural(model)
    elif attack_params["attack_type"] == 'fgsm':
        attack = FGSM(model, loss_fn, attack_params["epsilon"], clamp)
    elif attack_params["attack_type"] == 'pgd':
        # attack = PGD(model, loss_fn, eps, iter, step_size, random, clamp, 'inf', restart_num, beta, flip_grad_ratio)
        attack = PGD(model, loss_fn, attack_params, clamp, 'inf', flip_grad_ratio)
    else:
        raise NameError('No attack named %s' % attack_params["attack_type"])

    return attack


