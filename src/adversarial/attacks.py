from abc import ABC, abstractmethod
from adversarial.functional import *


class Attack(ABC):
    """Base class for adversarial attack methods"""
    @abstractmethod
    def create_adversarial_sample(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class FGSM(Attack):
    """Implements the Fast Gradient-Sign Method (FGSM).

    FGSM is a white box attack.
    clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST
    """
    def __init__(self,
                 model: Module,
                 loss_fn: Callable,
                 eps: float,
                 clamp: Tuple[float, float] = (0, 1)):
        super(FGSM, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.clamp = clamp

    def create_adversarial_sample(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor):
        """Creates an adversarial sample

        Args:
            x: Batch of samples
            y: Corresponding labels

        Returns:
            x_adv: Adversarially perturbed version of x
        """
        return fgsm(self.model, x, y, self.loss_fn, self.eps, self.clamp)


class PGD(Attack):
    """Implements the iterated Fast Gradient-Sign Method"""
    def __init__(self,
                 model: Module,
                 loss_fn: Callable,
                 eps: float,
                 k: int,
                 step: float,
                 clamp: Tuple[float, float] = (0, 1),
                 norm: Union[str, int] = 'inf'):
        super(PGD, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.step = step
        self.k = k
        self.norm = norm
        self.clamp = clamp

    def create_adversarial_sample(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor) -> torch.Tensor:
        return iterated_fgsm(self.model, x, y, self.loss_fn, self.k, self.step, self.eps, self.norm, random=True,
                             clamp=self.clamp)


class IteratedFGSM(Attack):
    """Implements the Projected Gradient Descent attack"""
    def __init__(self,
                 model: Module,
                 loss_fn: Callable,
                 eps: float,
                 k: int,
                 step: float,
                 norm: Union[str, int] = 'inf'):
        super(IteratedFGSM, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.step = step
        self.k = k
        self.norm = norm

    def create_adversarial_sample(self,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        return pgd(self.model, x, y, self.loss_fn, self.k, self.step, self.eps, self.norm, clamp)


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

