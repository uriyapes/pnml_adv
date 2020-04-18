# Code is based on https://github.com/oscarknagg/adversarial
from typing import Union
from abc import ABC, abstractmethod
from adversarial.functional import *
import torch
import os


class Adversarials(object):
    def __init__(self, attack_params: dict, original_sample: torch.Tensor, true_label: torch.Tensor, probability: torch.Tensor, loss: torch.Tensor,
                  adversarial_sample=None, genie_prob=None):
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
        self.original_sample = original_sample.cpu()
        self.true_label = true_label.cpu()
        self.adversarial_sample = adversarial_sample.cpu() if adversarial_sample is not None else None
        self.probability = probability.cpu()
        self.predict = torch.max(probability, 1)[1].cpu()
        self.correct = (self.predict == self.true_label)
        self.loss = loss.cpu()
        if genie_prob is not None:
            self.genie_prob = genie_prob.cpu()
            self.regret = torch.log(self.genie_prob.sum(dim=1, keepdim=False))
        else:
            self.genie_prob = None
            self.regret = None

    def merge(self, adv):
        self.original_sample = torch.cat([self.original_sample, adv.original_sample], dim=0)
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
                 eps_ratio: float,
                 k: int,
                 step: float,
                 random: bool,
                 clamp: Tuple[float, float] = (0, 1),
                 norm: Union[str, int] = 'inf',
                 restart_num: int = 1,
                 beta=0.0075,
                 flip_grad_ratio=0.0):
        super(PGD, self).__init__()
        self.name = self.__class__.__name__
        self.model = model
        self.loss_fn = loss_fn
        self.eps_ratio = eps_ratio
        self.k = k
        self.norm = norm
        self.random = random
        self.clamp = clamp
        self.step = step * (self.clamp[1] - self.clamp[0])
        self.eps = self.eps_ratio * (self.clamp[1] - self.clamp[0])
        self.restart_num = restart_num
        self.beta = beta
        self.flip_grad_ratio = flip_grad_ratio

    def create_adversarial_sample(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  y_target: torch.Tensor = None,
                                  get_adversarial_class: bool = False) -> Union[torch.Tensor, Adversarials]:
        adv_sample, adv_loss, adv_pred, genie_pred = iterated_fgsm(self.model, x, y, self.loss_fn, self.k, self.step, self.eps,
                                                       self.norm, y_target=y_target, random=self.random, clamp=self.clamp,
                                                       restart_num=self.restart_num, beta=self.beta, flip_grad_ratio=self.flip_grad_ratio)
        if get_adversarial_class:
            adversarials = Adversarials(self.__dict__, x, y, adv_pred, adv_loss, adv_sample, genie_pred)
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


def get_attack(attack_type: str, model: Module = None, eps: float = 0.3, iter: int = 30, step_size: float = 0.01,
               clamp: Tuple[float, float] = (0, 1), restart_num: int = 1, loss_fn: Callable = torch.nn.CrossEntropyLoss,
               beta=0.0, flip_grad_ratio=0.0):
    if restart_num == 0:
        random = False
        restart_num = 1
    else:
        random = True
    if attack_type == 'no_attack':
        attack = NoAttack()
    elif attack_type == 'fgsm':
        attack = FGSM(model, loss_fn, eps, clamp)
    elif attack_type == 'pgd':
        attack = PGD(model, loss_fn, eps, iter, step_size, random, clamp, 'inf', restart_num, beta, flip_grad_ratio)
    else:
        raise NameError('No attack named %s' % attack_type)

    return attack


