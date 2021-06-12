from abc import ABC, abstractmethod
from typing import Tuple, Union, Callable

import torch
from torch.nn import Module

from adversarial.adversarial_container import AdversarialContainer


class BaseAttack(ABC):
    def __init__(self,
                 model_to_attack: Module,
                 model_to_eval: Module,
                 loss_fn: Callable,
                 attack_params: dict,
                 clamp: Tuple[float, float] = (0, 1),
                 norm: Union[str, int] = 'inf'):
        self.name = self.__class__.__name__
        self.model_to_attack = model_to_attack
        self.model_to_eval = model_to_eval
        self.loss_fn = loss_fn  # TODO: not used currently
        self.attack_params = attack_params.copy()

        self.attack_params["norm"] = norm  # TODO: Should be inside the params
        self.attack_params["clamp"] = clamp

    def get_adversarial_tensor(self, x: torch.Tensor, y: Union[torch.Tensor, None], y_target: torch.Tensor = None) -> torch.Tensor:
        adv_sample, adv_loss, adv_pred, genie_pred = self.create_adversarial_sample(x, y, y_target)
        return adv_sample

    def get_adversarial_object(self, x: torch.Tensor, y: Union[torch.Tensor, None], y_target: torch.Tensor = None) -> AdversarialContainer:
        adv_sample, adv_loss, adv_pred, genie_pred = self.create_adversarial_sample(x, y, y_target)
        adv_sample = adv_sample if self.attack_params['save_adv_sample'] else None
        original_sample = x if self.attack_params['save_original_sample'] else None
        adversarials = AdversarialContainer(self.attack_params, original_sample, y, adv_pred, adv_loss, adv_sample, genie_pred)
        return adversarials

    def create_adversarial_sample(self, x: torch.Tensor, y: Union[torch.Tensor, None], y_target: torch.Tensor = None):
        self.model_to_attack.freeze_all_layers()
        if self.model_to_eval:
            self.model_to_eval.freeze_all_layers()

        adv_sample, adv_loss, adv_pred, genie_pred = self._create_adversarial_sample(x, y, y_target)

        self.model_to_attack.unfreeze_all_layers()
        if self.model_to_eval:
            self.model_to_eval.unfreeze_all_layers()

        return adv_sample, adv_loss, adv_pred, genie_pred

    @abstractmethod
    def _create_adversarial_sample(self,  x: torch.Tensor, y: Union[torch.Tensor, None], y_target: torch.Tensor = None):
        raise NotImplementedError
