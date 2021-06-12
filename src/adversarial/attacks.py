# Code is based on https://github.com/oscarknagg/adversarial
from abc import ABC, abstractmethod

from .adaptive_attack import EotPgdAttack
from .adversarial_container import AdversarialContainer
from .functional import *
import torch
from typing import Union, List


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
        assert(model)
        self.name = self.__class__.__name__
        self.model = model

    def create_adversarial_sample(self, x: torch.Tensor, y: torch.Tensor, y_target: torch.Tensor = None,
                                  get_adversarial_class: bool = False, save_adv_sample: bool = False,
                                  save_original_sample: bool = False):
        loss, prob, genie_prob = self.model.eval_batch(x, y, self.model.pnml_model)
        if get_adversarial_class:
            adv_sample = x if save_adv_sample else None
            # original_sample = x if save_original_sample else None  # Same as adv_sample
            return AdversarialContainer(None, adv_sample, None, y, prob, loss, adv_sample, genie_prob)
        else:
            return x


class FgsmRefine:
    """Implements the Fast Gradient-Sign Method (FGSM).

    FGSM is a white box attack.

    """
    def __init__(self,
                 model: Module,
                 loss_fn: Callable,
                 eps_ratio: float,
                 num_class, clamp: Tuple[float, float] = (0, 1)):
        """
        :param model: the model which will be used to create the adversarial examples, pay attention that as the model
        changes (by training) so is the adversarial examples created
        :param loss_fn: the loss function which the adversarial example tries to maximize
        :param eps_ratio: the L_infinity norm max value units of the total range of the allowed change (meaning that the
        epsilon value is adjusted for different images value ranges).
        :param: clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST
        """
        super(FgsmRefine, self).__init__()
        self.name = self.__class__.__name__
        self.model = model
        self.loss_fn = loss_fn()
        self.eps_ratio = eps_ratio
        self.num_class = num_class
        self.clamp = clamp
        self.eps = self.eps_ratio * (self.clamp[1] - self.clamp[0])

    def create_refined_samples(self, x: torch.Tensor):
        """Creates an adversarial sample

        Args:
            :param: x: Batch of samples

        Returns:
            x_adv: Adversarially perturbed version of x
        """
        return fgsm_all_labels(self.model, x, self.loss_fn, self.eps, self.clamp, class_num=self.num_class)


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
        if self.attack_params.get("beta") is None:
            self.attack_params["beta"] = 0.0
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
                                  y: Union[torch.Tensor, None],
                                  y_target: torch.Tensor = None,
                                  get_adversarial_class: bool = False,
                                  save_adv_sample: bool = True,
                                  save_original_sample: bool = False) -> Union[torch.Tensor, Adversarials]:
        adv_sample, adv_loss, adv_pred, genie_pred = iterated_fgsm(
                self.model, x, y, self.loss_fn, self.attack_params["pgd_iter"],self.attack_params["pgd_step"],
                self.attack_params["epsilon"], self.attack_params["norm"], y_target=y_target, random=self.attack_params["random"],
                clamp=self.attack_params["clamp"], restart_num=self.attack_params["pgd_test_restart_num"],
                beta=self.attack_params["beta"], flip_grad_ratio=self.attack_params["flip_grad_ratio"])
        if get_adversarial_class:
            adv_sample = adv_sample if save_adv_sample else None
            original_sample = x if save_original_sample else None
            adversarials = Adversarials(self.attack_params, original_sample, y, adv_pred, adv_loss, adv_sample, genie_pred)
            return adversarials
        else:
            return adv_sample

# from cleverhans.future.torch.attacks.spsa import spsa
import logger_utilities

class SPSA(Attack):
    def __init__(self, model: Module, loss_fn: Callable, attack_params: dict,clamp: Tuple[float, float] = (0, 1)):
        self.model = model
        self.attack_params = attack_params
        self.clamp = clamp

    def create_adversarial_sample(self, x: torch.Tensor, y: Union[torch.Tensor], get_adversarial_class: bool = False,
                                  save_adv_sample: bool = True, save_original_sample: bool = False) -> Union[torch.Tensor, AdversarialContainer]:
        # The true batch size (the number of evaluated inputs for each update) is `spsa_samples * spsa_iters
        logger = logger_utilities.get_logger()
        pnml_shrink_batch_factor = 1
        if self.model.pnml_model is True:
            spsa_samples = int(self.attack_params["spsa_samples"]/(pnml_shrink_batch_factor*self.attack_params["evals_per_update"]))
            evals_per_update = int(self.attack_params["evals_per_update"] * pnml_shrink_batch_factor)
            assert(spsa_samples>0)
            assert(evals_per_update>0)
            model_fn = lambda x: torch.log(self.model.calc_logits(x))
            early_stop_loss_threshold = (-0.01, 4)
        else:
            spsa_samples = self.attack_params["spsa_samples"]
            evals_per_update = self.attack_params["evals_per_update"]
            model_fn = self.model
            early_stop_loss_threshold = (-0.01, 10)
        with torch.set_grad_enabled(self.model.pnml_model):
            adv_sample = spsa(model_fn, x, self.attack_params["epsilon"], self.attack_params["pgd_iter"], y=y, norm=np.inf,
                 clip_min=self.clamp[0], clip_max=self.clamp[1], learning_rate=0.01, delta=0.01, spsa_samples=spsa_samples,
                      spsa_iters= evals_per_update, early_stop_loss_threshold=early_stop_loss_threshold, is_debug=False, sanity_checks=False)
        if get_adversarial_class:
            adv_loss, prob, genie_prob = self.model.eval_batch(adv_sample, y, enable_grad=self.model.pnml_model)

            adv_sample = adv_sample if save_adv_sample else None
            original_sample = x if save_original_sample else None
            adversarials = AdversarialContainer(self.attack_params, original_sample, y, prob, adv_loss, adv_sample, genie_prob)
            logger.info("SPSA adversarial sample is correct{}".format(adversarials.correct.item()))
            return adversarials
        else:
            return adv_sample


class PgdRefine(PGD):
    def __init__(self, num_class: int, *args, **kwargs):
        super(PgdRefine, self).__init__(*args, **kwargs)
        self.num_class = num_class

    def create_refined_samples(self, x: torch.Tensor):
        # batch_size = x.shape[0]
        # labels_mat = torch.arange(0, self.num_class, dtype=torch.long, device=x.device).unsqueeze(dim=0).expand(batch_size, self.num_class)
        x_adv = torch.zeros_like(x, device=x.device).unsqueeze(dim=0).repeat(
            [self.num_class] + [1 for i in range(x.dim())])
        for label in range(self.num_class):
            torch_label = torch.ones([x.shape[0]], dtype=torch.long).to(x.device) * label
            x_adv[label] = super(PgdRefine, self).create_adversarial_sample(x, None, torch_label).detach()
        return x_adv


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


def get_attack(attack_params: dict, model_to_attack: Module = None, model_to_eval: Module = None, clamp: Tuple[float, float] = (0, 1),
               loss_fn: Callable = torch.nn.CrossEntropyLoss, flip_grad_ratio=0.0, num_class:int = 10):
    """
    :param attack_params: a dictionary containing attack parameters. Dictionary keys:
        attack_type
        epsilon
        pgd_iter
        pgd_step
        pgd_test_restart_num
        beta
    :param model_to_attack: The model_to_attack to be attacked
    :param clamp: The value range of the samples
    :param  loss_fn:
    :param flip_grad_ratio:
    """
    if attack_params["attack_type"] == 'no_attack':
        attack = NoAttack()
    elif attack_params["attack_type"] == "natural":
        attack = Natural(model_to_attack)
    elif attack_params["attack_type"] == 'fgsm':
        raise DeprecationWarning("For fgsm attack use pgd with pgd_test_restart_num=0, pgd_iter=1, pgd_step=epsilon")
        # attack = FGSM(model_to_attack, loss_fn, attack_params["epsilon"], num_class, clamp)
    elif attack_params["attack_type"] == 'pgd':
        # attack = PGD(model_to_attack, loss_fn, eps, iter, step_size, random, clamp, 'inf', restart_num, beta, flip_grad_ratio)
        attack = PGD(model_to_attack, model_to_eval, loss_fn, attack_params, clamp, 'inf', flip_grad_ratio)
    elif attack_params["attack_type"] == 'eot':
        attack = EotPgdAttack(model_to_attack, model_to_eval, loss_fn, attack_params, clamp, 'inf')
    elif attack_params["attack_type"] == 'spsa':
        attack = SPSA(model_to_attack, loss_fn, attack_params, clamp)
    else:
        raise NameError('No attack named %s' % attack_params["attack_type"])

    return attack


def get_refiner(attack_params: dict, model: Module = None, clamp: Tuple[float, float] = (0, 1),
               loss_fn: Callable = torch.nn.CrossEntropyLoss, flip_grad_ratio=0.0, num_class:int = 10):
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
    attack_params["beta"] = 0.0
    if attack_params["attack_type"] == 'fgsm':
        refine = FgsmRefine(model, loss_fn, attack_params["epsilon"], num_class, clamp)
    elif attack_params["attack_type"] == 'pgd':
        refine = PgdRefine(num_class, model, loss_fn, attack_params, clamp, 'inf', flip_grad_ratio)
    else:
        raise NameError('No attack named %s' % attack_params["attack_type"])
    return refine


