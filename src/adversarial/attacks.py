# Code is based on https://github.com/oscarknagg/adversarial
from abc import ABC, abstractmethod
from adversarial.functional import *


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
                 restart_num: int = 1):
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

    def create_adversarial_sample(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  y_target: torch.Tensor = None) -> torch.Tensor:
        return iterated_fgsm(self.model, x, y, self.loss_fn, self.k, self.step, self.eps, self.norm, y_target=y_target,
                             random=self.random, clamp=self.clamp, restart_num=self.restart_num)


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
                random: bool = True, clamp: Tuple[float, float] = (0, 1), restart_num: int = 1,
                loss_fn: Callable = torch.nn.CrossEntropyLoss):
    if attack_type == 'no_attack':
        attack = NoAttack()
    elif attack_type == 'fgsm':
        attack = FGSM(model, loss_fn, eps, clamp)
    elif attack_type == 'pgd':
        attack = PGD(model, loss_fn, eps, iter, step_size, random, clamp, 'inf', restart_num)
    else:
        raise NameError('No attack named %s' % attack_type)

    return attack


