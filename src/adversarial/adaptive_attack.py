import gc
import time
from typing import Tuple, Union, Callable
import numpy as np
import torch
from adversarial.base_attack import BaseAttack
from adversarial.utils import add_uniform_random_noise, project
from utilities import TorchUtils


class EotPgdAttack(BaseAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.attack_params.get("beta") is None:
            self.attack_params["beta"] = 0.0
        if self.attack_params["pgd_test_restart_num"] == 0:
            self.attack_params["random"] = False
            self.attack_params["pgd_test_restart_num"] = 1
        else:
            self.attack_params["random"] = True
        assert ((self.attack_params["random"] is False and self.attack_params["pgd_test_restart_num"] == 1)
                or (self.attack_params["random"] is True and self.attack_params["pgd_test_restart_num"] >= 1))
        self.attack_params["epsilon"] = self.attack_params["epsilon"] * (self.attack_params["clamp"][1] - self.attack_params["clamp"][0])
        self.attack_params["pgd_step"] = self.attack_params["pgd_step"] * (self.attack_params["clamp"][1] - self.attack_params["clamp"][0])
        self.loss_fn = torch.nn.NLLLoss(reduction='none')
        if self.attack_params["EOT_num_of_iter_to_avg"] == 0:
            self.attack_params["perform_EOT"] = False
            self.attack_params["EOT_num_of_iter_to_avg"] = np.inf
        else:
            self.attack_params["perform_EOT"] = True
        # self.is_record_stats = is_record_stats
        # self.choose_best_adv_recorded = choose_best_adv_recorded
        # assert (self.choose_best_adv_recorded and self.is_record_stats) or self.choose_best_adv_recorded is False
        self.iter_to_clear_buff = 300

    def _create_adversarial_sample(self, x: torch.Tensor, y: Union[torch.Tensor, None], y_target: torch.Tensor = None):
        x_adv_l = []
        loss_l = []
        prediction_l = []
        genie_prob_l = []
        loss_per_iter_l = []
        # We want to get the element-wise loss to decide which sample has the highest loss compared to the other random
        # start. Make sure the loss_fn that was received is cross-entropy
        for i in range(self.attack_params["pgd_test_restart_num"]):
            if self.attack_params["random"]:
                x_adv = add_uniform_random_noise(x, self.attack_params["epsilon"], self.attack_params["clamp"]).detach()
            else:
                x_adv = x.detach().clone().to(x.device)

            x_adv, loss, prediction, genie_prob, loss_per_iter = self._pgd_attack(x, x_adv, y, y_target)
            x_adv_l.append(x_adv)
            loss_l.append(loss)
            prediction_l.append(prediction)
            genie_prob_l.append(genie_prob)
            loss_per_iter_l.append(loss_per_iter)
            # print("loss in iter{}:".format(i) + str(loss))

        if self.attack_params["pgd_test_restart_num"] == 1:  # TODO: This isn't needed
            chosen_adv = x_adv_l[0]
            chosen_loss = loss_l[0]
            chosen_prediction = prediction_l[0]
            chosen_genie_prob = genie_prob_l[0]
            chosen_loss_per_iter = loss_per_iter_l[0]
        else:
            x_adv_stack = torch.stack(x_adv_l)  # TODO: allocate memory in advance to speed up
            loss_stack = torch.stack(loss_l)
            prediction_stack = torch.stack(prediction_l)
            if y_target is None:
                best_loss_ind = torch.argmax(loss_stack, dim=0).tolist()  # find the maximum loss between all the random starts
            else:
                best_loss_ind = torch.argmin(loss_stack, dim=0).tolist()  # find the minimum loss for the specified y_target
            chosen_adv = x_adv_stack[best_loss_ind, range(x_adv_stack.size()[1])]  # make max_loss_ind numpy
            chosen_loss = loss_stack[best_loss_ind, range(x_adv_stack.size()[1])]
            chosen_prediction = prediction_stack[best_loss_ind, range(x_adv_stack.size()[1])]
            chosen_genie_prob = torch.stack(genie_prob_l)[best_loss_ind, range(x_adv_stack.size()[1])] if genie_prob_l[0] is not None else None
            chosen_loss_per_iter = torch.stack(loss_per_iter_l)[best_loss_ind, range(x_adv_stack.size()[1])]
        return chosen_adv, chosen_loss, chosen_prediction, chosen_genie_prob, chosen_loss_per_iter

    def _pgd_attack(self,  x: torch.Tensor, x_adv: torch.Tensor, y: Union[torch.Tensor, None], y_target: torch.Tensor = None):
        # Prepare data:
        self._is_targeted = y_target is not None
        self._init_record_stats_buffers(x)

        for i in range(self.attack_params["pgd_iter"] + 1):
            # t0 = time.time()
            x_adv = self._pgd_iteration(x, x_adv, i, y_target, y)
            # print("_iterative_gradient iter: {}, time passed: {}, max CUDA memory: {}".format(i, time.time() - t0,
            #                                                                  torch.cuda.max_memory_allocated() / 2 ** 20))

        #  This is done so model with refinement could do backprop
        # x_adv = x_adv.detach()
        # x_adv.requires_grad_(True)

        if self.model_to_eval is not None:
            loss_best, prob, genie_prob = self.model_to_eval.eval_batch(self.x_adv_best, y_target if self._is_targeted else y,
                                                       enable_grad=self.model_to_eval.pnml_model)
        else:
            loss_best, prob, genie_prob = self.model_to_attack.eval_batch(self.x_adv_best, y_target if self._is_targeted else y,
                                                       enable_grad=self.model_to_attack.pnml_model)

        return self.x_adv_best, loss_best.detach(), prob.detach(), genie_prob, self.loss_per_iter.T

    def _pgd_iteration(self, x, x_adv, i, y_target, y):
        x_adv.detach_()
        x_adv = x_adv.requires_grad_(True)
        # prediction = self.model_to_attack.calc_log_prob(x_adv)
        # loss = self.loss_fn(prediction, y_target if self._is_targeted else y)
        loss = self._get_model_loss(self.model_to_attack, x_adv, y, y_target)
        loss_mean = loss.mean() - self.attack_params["beta"] * self.model_to_attack.regularization.mean()

        # Record x_adv and corresponding loss:
        if self.model_to_eval is not None:
            loss_to_record, _, _ = self.model_to_eval.eval_batch(x_adv, y_target if self._is_targeted else y,
                                                                 enable_grad=self.model_to_eval.pnml_model)
        else:
            loss_to_record = loss.detach()
        self._record_stats(x_adv, i, loss_to_record)

        # In last iteration there is no need to calc the next x_adv so break loop
        if i == (self.attack_params["pgd_iter"]):
            return
        if (i + 1) % self.iter_to_clear_buff == 0:
            gc.collect()
            if TorchUtils.get_device() == "cuda":
                torch.cuda.empty_cache()

        x_adv_grad = torch.autograd.grad(loss_mean, x_adv, create_graph=False)[0]
        if self.attack_params["perform_EOT"]:
            self._record_EOT_variables(x_adv_grad, i % self.attack_params["EOT_num_of_iter_to_avg"])

        # Add perturbation and project back into l_norm ball around x:
        if (i + 1) % self.attack_params["EOT_num_of_iter_to_avg"] != 0:
            pert = self._calc_perturbation(x_adv, x_adv_grad)
            x_adv = project(x, x_adv + pert, self.attack_params["norm"], self.attack_params["epsilon"]).clamp(
                *self.attack_params["clamp"]).detach_()
        else:
            with torch.no_grad():
                mean_grad = self.grad_per_iter.mean(dim=0)
                pert = self._calc_perturbation(x_adv, mean_grad)
                x_adv = project(x, self.pre_EOT_x_adv + pert, self.attack_params["norm"],
                                self.attack_params["epsilon"]).clamp(*self.attack_params["clamp"]).detach_()
                self.pre_EOT_x_adv = x_adv
        return x_adv

    def _record_stats(self, x_adv, idx, loss):
        # Update statistics for current iteration
        self.loss_per_iter[idx, :] = loss.detach()
        update_loss_max_idx = (self.loss_best < self.loss_per_iter[idx, :]).detach()
        self.loss_best[update_loss_max_idx] = self.loss_per_iter[idx, update_loss_max_idx]
        self.x_adv_best[update_loss_max_idx] = x_adv[update_loss_max_idx].detach()

    # noinspection PyPep8Naming
    def _record_EOT_variables(self, x_adv_grad, idx):
        self.grad_per_iter[idx, :] = x_adv_grad.detach()

    def _init_record_stats_buffers(self, x):
        self.x_adv_best = torch.zeros_like(x).detach()
        self.loss_best = -1 * torch.ones(x.shape[0], dtype=torch.float32,device=x.device)  # Init the loss to -1 so in first iteration the loss_best and x_adv_best will be updated
        self.loss_per_iter = torch.zeros([self.attack_params["pgd_iter"] + 1, x.shape[0]], dtype=torch.float32, device=x.device)
        if self.attack_params["perform_EOT"]:
            self._init_record_EOT_buffers(x)

    # noinspection PyPep8Naming
    def _init_record_EOT_buffers(self, x):
        self.grad_per_iter = torch.zeros([self.attack_params["EOT_num_of_iter_to_avg"]] + list(x.shape), dtype=torch.float32,
                                         device=x.device)
        self.pre_EOT_x_adv = torch.zeros_like(x).detach()

    def _calc_perturbation(self, x_adv, x_adv_grad) -> torch.Tensor:
        gradient_dir = -1 if self._is_targeted else 1  # Gradient descent for targeted attack and gradient ascent for untargeted attack
        with torch.no_grad():
            if self.attack_params["norm"] == 'inf':
                return (gradient_dir * x_adv_grad.sign() * self.attack_params["pgd_step"]).detach()
            else:
                # .view() assumes batched image data as 4D tensor
                return gradient_dir * x_adv_grad * self.attack_params["pgd_step"] / x_adv_grad.view(x_adv.shape[0], -1). \
                    norm(self.attack_params["norm"], dim=-1).view(-1, 1, 1, 1).detach()

    def _get_model_loss(self, model, x_adv, y, y_target):
        prediction_on_model_to_eval = model.calc_log_prob(x_adv)
        return self.loss_fn(prediction_on_model_to_eval, y_target if self._is_targeted else y)


