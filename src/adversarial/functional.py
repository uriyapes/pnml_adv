from typing import Union, Callable, Tuple
from functools import reduce
from collections import deque
from torch.nn import Module
import torch
import numpy as np
from adversarial.utils import project, generate_misclassified_sample, random_perturbation


def fgsm(model: Module,
         x: torch.Tensor,
         y: torch.Tensor,
         loss_fn: Callable,
         eps: float,
         y_target = None,
         clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Creates an adversarial sample using the Fast Gradient-Sign Method (FGSM)

    This is a white-box attack.

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        eps: Size of adversarial perturbation
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    x = x.detach()
    x.requires_grad = True
    targeted = y_target is not None
    prediction = model(x, y)
    loss = loss_fn(prediction, y_target if targeted else y)
    loss.backward(retain_graph=True)

    # x_adv = (x + torch.sign(x.grad) * eps).clamp(*clamp).detach()
    # x_adv = (x + x.grad + torch.sign(x.grad) * eps).detach()
    # x_grad_sign = 1.0/100 * x.grad.sign()
    x_grad_sign = x.grad *500
    if not targeted:
        x_adv = (x + x_grad_sign * eps).detach()#.clamp(*clamp)
    else:
        x_adv = (x - x_grad_sign * eps).detach()#.clamp(*clamp)
    x.requires_grad = False
    return x_adv


def _iterative_gradient(model: Module,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        loss_fn: Callable,
                        k: int,
                        step: float,
                        eps: float,
                        norm: Union[str, float],
                        step_norm: Union[str, float],
                        y_target: torch.Tensor = None,
                        random: bool = False,
                        clamp: Tuple[float, float] = (0, 1),
                        beta=0.0) -> Tuple[torch.Tensor, int]:
    """Base function for PGD and iterated FGSM

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        k: Number of iterations to make
        step: Size of step to make at each iteration
        eps: Maximum size of adversarial perturbation, larger perturbations will be projected back into the
            L_norm ball
        norm: Type of norm
        step_norm: 'inf' for PGD, 2 for iterated PGD (MEK: corrected)
        y_target:
        random: Whether to start Iterated FGSM within a random point in the l_norm ball
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    targeted = y_target is not None
    # x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    x_adv = x.clone().to(x.device)
    if random:
        # x_adv = random_perturbation(x_adv, norm, eps)
        rand_gen = torch.distributions.uniform.Uniform(x_adv - eps, x_adv + eps)  #Create a point around x_adv within a range of eps
        x_adv = rand_gen.sample().clamp(*clamp)

    for i in range(k):
        # print("_iterative_gradient iter: {}".format(i))
        # Each new loop x_adv is a new variable (x_adv += gradients), therefore we must detach it (otherwise backward()
        # will result in calculating the old clones gradients as well) and then requires_grad_(True) since detach()
        # disabled the grad.
        # The other option (original) is to work with temp variable _x_adv (see below) but it seems to prelong the
        # calculation time maybe as a result of re-cloning
        # _x_adv = x_adv.clone().detach().requires_grad_(True)
        x_adv = x_adv
        x_adv.requires_grad_(True)
        prediction = model(x_adv, y)
        loss = loss_fn(prediction, y_target if targeted else y).mean() - beta*torch.log(model.regularization.mean())#+ 0.5 * model.regularization.mean() TODO: uncomment
        # loss.backward()
        # x_adv_grad = x_adv.grad
        x_adv_grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        with torch.no_grad():
            if step_norm == 'inf':
                gradients = (x_adv_grad.sign() * step).detach()
            else:
                # .view() assumes batched image data as 4D tensor
                gradients = x_adv_grad * step / x_adv_grad.view(x_adv.shape[0], -1).norm(step_norm, dim=-1)\
                    .view(-1, 1, 1, 1)

            if targeted:
                # Targeted: Gradient descent on the loss of the (incorrect) target label
                # w.r.t. the model parameters (increasing prob. to predict the incorrect label)
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += gradients


        # Project back into l_norm ball and correct range
        x_adv = project(x, x_adv, norm, eps).clamp(*clamp)
    x_adv = x_adv
    # x_adv.requires_grad_(True) #  This is done so model with refinement could do backprop
    prediction = model(x_adv, y)
    adv_loss = loss_fn(prediction, y_target if targeted else y)
    x_adv.requires_grad_ = False

    return x_adv, adv_loss


def iterated_fgsm(model: Module,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  loss_fn: Callable,
                  k: int,
                  step: float,
                  eps: float,
                  norm: Union[str, float],
                  y_target: torch.Tensor = None,
                  random: bool = False,
                  clamp: Tuple[float, float] = (0, 1),
                  restart_num: int = 1,
                  beta = 0.0075) -> torch.Tensor:
    """Creates an adversarial sample using the iterated Fast Gradient-Sign Method

    This is a white-box attack.

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        k: Number of iterations to make
        step: Size of step to make at each iteration
        eps: Maximum size of adversarial perturbation, larger perturbations will be projected back into the
            L_norm ball
        norm: Type of norm
        y_target:
        random: Whether to start Iterated FGSM within a random point in the l_norm ball
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST
        restart_num: the number of random restarts to attempt to locate the best adversary

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    assert((random is False and restart_num == 1) or (random is True and restart_num >= 1))
    # is_model_training_flag = model.training
    # model.train(False)  # The same as model.eval()
    model.freeze_all_layers()

    max_loss = -1
    x_adv_l = []
    loss_l = []
    # We want to get the element-wise loss to decide which sample has the highest loss compared to the other random
    # start. Make sure the loss_fn that was received is cross-entropy
    loss_fn = loss_fn(reduction='none')
    for i in range(restart_num):
        x_adv, loss = _iterative_gradient(model=model, x=x, y=y, loss_fn=loss_fn, k=k, eps=eps, norm=norm, step=step,
                                   step_norm='inf', y_target=y_target, random=random, clamp=clamp, beta=beta)
        x_adv_l.append(x_adv)
        loss_l.append(loss)
        # print("loss in iter{}:".format(i) + str(loss))
        # if loss > max_loss:
        #     best_adv = x_adv

    if restart_num == 1:
        best_adv = x_adv_l[0]
    else:
        x_adv_stack = torch.stack(x_adv_l)
        loss_stack = torch.stack(loss_l)
        if y_target is None:
            best_loss_ind = torch.argmax(loss_stack, dim=0).tolist()  # find the maximum loss between all the random starts
        else:
            best_loss_ind = torch.argmin(loss_stack, dim=0).tolist()  # find the minimum loss for the specified y_target
        best_adv = x_adv_stack[best_loss_ind, range(x_adv_stack.size()[1])]  # make max_loss_ind numpy
    model.unfreeze_all_layers()
    # model.train(is_model_training_flag)
    return best_adv


def boundary(model: Module,
             x: torch.Tensor,
             y: torch.Tensor,
             k: int,
             orthogonal_step: float = 1e-2,
             perpendicular_step: float = 1e-2,
             initial: torch.Tensor = None,
             clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Implements the boundary attack

    This is a black box attack that doesn't require knowledge of the model
    structure. It only requires knowledge of

    https://arxiv.org/pdf/1712.04248.pdf

    Args:
        model: Model to be attacked
        x: Batched image data
        y: Corresponding labels
        k: Number of steps to take
        orthogonal_step: orthogonal step size (delta in paper)
        perpendicular_step: perpendicular step size (epsilon in paper)
        initial: Initial attack image to start with. If this is None then use random noise
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Best i.e. closest adversarial example for x
    """
    orth_step_stats = deque(maxlen=30)
    perp_step_stats = deque(maxlen=30)
    # Factors to adjust step sizes by
    orth_step_factor = 0.97
    perp_step_factor = 0.97

    def _propose(x: torch.Tensor,
                 x_adv: torch.Tensor,
                 y: torch.Tensor,
                 model: Module,
                 clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        """Generate proposal perturbed sample

        Args:
            x: Original sample
            x_adv: Adversarial sample
            y: Label of original sample
            clamp: Domain (i.e. max/min) of samples
        """
        # Sample from unit Normal distribution with same shape as input
        perturbation = torch.normal(torch.zeros_like(x_adv), torch.ones_like(x_adv))

        # Rescale perturbation so l2 norm is delta
        perturbation = project(torch.zeros_like(perturbation), perturbation, norm=2, eps=orthogonal_step)

        # Apply perturbation and project onto sphere around original sample such that the distance
        # between the perturbed adversarial sample and the original sample is the same as the
        # distance between the unperturbed adversarial sample and the original sample
        # i.e. d(x_adv, x) = d(x_adv + perturbation, x)
        perturbed = x_adv + perturbation
        perturbed = project(x, perturbed, 2, torch.norm(x_adv - x, 2)).clamp(*clamp)

        # Record success/failure of orthogonal step
        orth_step_stats.append(model(perturbed).argmax(dim=1) != y)

        # Make step towards original sample
        step_towards_original = project(torch.zeros_like(perturbation), x - perturbed, norm=2, eps=perpendicular_step)
        perturbed = (perturbed + step_towards_original).clamp(*clamp)

        # Record success/failure of perpendicular step
        perp_step_stats.append(model(perturbed).argmax(dim=1) != y)

        # Clamp to domain of sample
        perturbed = perturbed.clamp(*clamp)

        return perturbed

    if x.size(0) != 1:
        # TODO: Attack a whole batch in parallel
        raise NotImplementedError

    if initial is not None:
        x_adv = initial
    else:
        # Generate initial adversarial sample from uniform distribution
        x_adv = generate_misclassified_sample(model, x, y)

    total_stats = torch.zeros(k)

    for i in range(k):
        # Propose perturbation
        perturbed = _propose(x, x_adv, y, model, clamp)

        # Check if perturbed input is adversarial i.e. gives the wrong prediction
        perturbed_prediction = model(perturbed).argmax(dim=1)
        total_stats[i] = perturbed_prediction != y
        if perturbed_prediction != y:
            x_adv = perturbed

        # Check statistics and adjust step sizes
        if len(perp_step_stats) == perp_step_stats.maxlen:
            if torch.Tensor(perp_step_stats).mean() > 0.5:
                perpendicular_step /= perp_step_factor
                orthogonal_step /= orth_step_factor
            elif torch.Tensor(perp_step_stats).mean() < 0.2:
                perpendicular_step *= perp_step_factor
                orthogonal_step *= orth_step_factor

        if len(orth_step_stats) == orth_step_stats.maxlen:
            if torch.Tensor(orth_step_stats).mean() > 0.5:
                orthogonal_step /= orth_step_factor
            elif torch.Tensor(orth_step_stats).mean() < 0.2:
                orthogonal_step *= orth_step_factor

    return x_adv


def _perturb(x: torch.Tensor,
             p: float,
             i: int,
             j: int,
             clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Perturbs a pixel in an image

    Args:
        x: image
        p: perturbation parameters
        i: row
        j: column
    """
    if x.size(0) != 1:
        raise NotImplementedError('Only implemented for single image')

    x[0, :, i, j] = p * torch.sign(x[0, :, i, j])

    return x.clamp(*clamp)


def local_search(model: Module,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 k: int,
                 branching: Union[int, float] = 0.1,
                 p: float = 1.,
                 d: int = None,
                 clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Performs the local search attack

    This is a black-box (score based) attack first described in
    https://arxiv.org/pdf/1612.06299.pdf

    Args:
        model: Model to attack
        x: Batched image data
        y: Corresponding labels
        k: Number of rounds of local search to perform
        branching: Either fraction of image pixels to search at each round or
            number of image pixels to search at each round
        p: Size of perturbation
        d: Neighbourhood square half side length

    Returns:
        x_adv: Adversarial version of x
    """
    if x.size(0) != 1:
        # TODO: Attack a whole batch at a time
        raise NotImplementedError('Only implemented for single image')

    x_adv = x.clone().detach().requires_grad_(False).to(x.device)
    model.eval()

    data_shape = x_adv.shape[2:]
    if isinstance(branching, float):
        branching = int(reduce(lambda x, y: x*y, data_shape) * branching)

    for _ in range(k):
        # Select pixel locations at random
        perturb_pixels = torch.randperm(reduce(lambda x, y: x*y, data_shape))[:branching]

        perturb_pixels = torch.stack([perturb_pixels // data_shape[0], perturb_pixels % data_shape[1]]).transpose(1, 0)

        # Kinda hacky but works for MNIST (i.e. 1 channel images)
        # TODO: multi channel images
        neighbourhood = x_adv.repeat((branching, 1, 1, 1))
        perturb_pixels = torch.cat([torch.arange(branching).unsqueeze(-1), perturb_pixels], dim=1)
        neighbourhood[perturb_pixels[:, 0], 0, perturb_pixels[:, 1], perturb_pixels[:, 2]] = 1

        predictions = model(neighbourhood).softmax(dim=1)
        scores = predictions[:, y]

        # Select best perturbation and continue
        i_best, j_best = perturb_pixels[scores.argmin(dim=0).item(), 1:]
        x_adv[0, :, i_best, j_best] = 1.
        x_adv.clamp_(*clamp)

        # Early exit if adversarial is found
        worst_prediction = predictions.argmax(dim=1)[scores.argmin(dim=0).item()]
        if worst_prediction.item() != y.item():
            return x_adv

    # Attack failed, return sample with lowest score of correct class
    return x_adv
