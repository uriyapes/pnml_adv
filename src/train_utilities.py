import os
import logging
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from adversarial.attacks import get_attack
from dataset_utilities import insert_sample_to_dataset, get_dataset_min_max_val
from utilities import TorchUtils


class TrainClass:
    """
    Class which execute train on a DNN model.
    """
    criterion = nn.CrossEntropyLoss()
    def __init__(self, params_to_train, learning_rate: float, momentum: float, step_size: list, gamma: float,
                 weight_decay: float, logger=None, adv_learn_alpha=0, adv_learn_eps=0.05, attack_type: str = 'pgd',
                 pgd_iter: int = 30, pgd_step: float = 0.01, pgd_random: bool = True, save_model_every_n_epoch = float('Inf')):
        """
        Initialize train class object.
        :param params_to_train: the parameters of pytorch Module that will be trained.
        :param learning_rate: initial learning rate for the optimizer.
        :param momentum:  initial momentum rate for the optimizer.
        :param step_size: reducing the learning rate by gamma each step_size.
        :param gamma:  reducing the learning rate by multiplicative of gamma each step_size.
        :param weight_decay: L2 regularization.
        :param logger: logger class in order to print logs and save results.
        :param adv_learn_alpha: The weight that should be given to the adversarial learning regulaizer (0 means none).
        :param attack_type: options: fgsm, pgd and none
        """

        self.logger = logger if logger is not None else logging.StreamHandler(sys.stdout)
        self.eval_test_during_train = True
        self.eval_test_in_end = True
        self.print_during_train = True
        self.save_model_every_n_epoch = save_model_every_n_epoch

        # Optimizer
        self.optimizer = optim.SGD(params_to_train,
                                   lr=learning_rate,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=step_size,
                                                        gamma=gamma)
        self.freeze_batch_norm = False
        self.adv_learn_alpha = adv_learn_alpha
        self.adv_learn_eps = adv_learn_eps
        self.attack_type = attack_type
        self.pgd_iter = pgd_iter
        self.pgd_step = pgd_step
        self.pgd_random = pgd_random

    def train_model(self, model, dataloaders, num_epochs: int = 10, acc_goal=None, eval_test_every_n_epoch: int = 1,
                    sample_test_data=None, sample_test_true_label=None):
        """
        Train DNN model using some trainset.
        :param model: the model which will be trained.
        :param dataloaders: contains the trainset for training and testset for evaluation.
        :param num_epochs: number of epochs to train the model.
        :param acc_goal: stop training when getting to this accuracy rate on the trainset.
        :return: trained model (also the training of the models happen inplace)
                 and the loss of the trainset and testset.
        """
        self.logger.info("Use device:" + TorchUtils.get_device())
        model = TorchUtils.to_device(model)
        attack = get_attack(self.attack_type, model, self.adv_learn_eps, self.pgd_iter, self.pgd_step,
                            self.pgd_random, get_dataset_min_max_val(dataloaders['dataset_name']))

        # If testset is already adversarial then do nothing else use the same attack to generate adversarial testset
        testset_attack = get_attack("no_attack") if dataloaders['adv_test_flag'] else attack  # TODO: replace training attack with testing attack
        epoch_time = 0

        # Loop on epochs
        for epoch in range(1,num_epochs+1):

            epoch_start_time = time.time()
            total_loss_in_epoch, natural_train_loss, train_acc = self.__train(model, dataloaders['train'], attack,
                                                                      sample_test_data, sample_test_true_label)
            # Evaluate testset
            if self.eval_test_during_train is True and epoch % eval_test_every_n_epoch == 0:
                test_loss, test_acc = self.eval_model(model, dataloaders['test'], testset_attack)
            else:
                test_loss, test_acc = torch.tensor([-1.]), torch.tensor([-1.])
            epoch_time = time.time() - epoch_start_time

            # Save model
            if epoch % self.save_model_every_n_epoch == 0:
                torch.save(model.state_dict(), os.path.join(self.logger.output_folder, 'model_iter_%d.pt' % epoch))
            # Log
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
            self.logger.info('[%d/%d] [train test] loss =[%f %f] natural_train_loss=[%f], acc=[%f %f], lr=%f, epoch_time=%.2f'
                             % (epoch, num_epochs,
                                total_loss_in_epoch, test_loss, natural_train_loss, train_acc, test_acc,
                                lr, epoch_time))

            # Stop training if desired goal is achieved
            if acc_goal is not None and train_acc >= acc_goal:
                break

        test_loss, test_acc = self.eval_model(model, dataloaders['test'], testset_attack)
        train_loss_output = float(total_loss_in_epoch.cpu().detach().numpy().round(16))
        test_loss_output = float(test_loss.cpu().detach().numpy().round(16))
        # Print and save
        self.logger.info('----- [train test] loss =[%f %f], natural_train_loss=[%f], acc=[%f %f] epoch_time=%.2f' %
                         (total_loss_in_epoch, test_loss, natural_train_loss, train_acc, test_acc,
                          epoch_time))

        return model, train_loss_output, test_loss_output

    def __train(self, model, train_loader, attack, sample_test_data=None, sample_test_true_label=None):
        """
        Execute one epoch of training
        :param model: the model that will be trained.
        :param train_loader: contains the trainset to train.
        :return: the loss and accuracy of the trainset.
        """
        model.train()

        # Turn off batch normalization update
        if self.freeze_batch_norm is True: # this works during fine-tuning because it can change the model even if LR=0.
            model = model.apply(set_bn_eval) # this fucntion calls model.eval() which only effects dropout and batchnorm which will work in eval mode.
        total_loss_in_epoch = 0
        correct = 0

        natural_loss = 0
        natural_train_loss_in_ep = 0 if self.adv_learn_alpha != 1 else -1*len(train_loader.dataset)
        natural_correct = 0

        loss_sample_test = 0
        # max_iter = np.ceil(len(train_loader.dataset) / train_loader.batch_size) #TODO: from some reason I am missing the last batch but the dataloader should not drop the last batch
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sample_test_data is not None:

            sample_test_data, sample_test_true_label = TorchUtils.to_device(sample_test_data),TorchUtils.to_device(sample_test_true_label)
            sample_test_data.requires_grad = False
            sample_test_true_label.requires_grad = False

        # Iterate over dataloaders
        self.logger.init_debug_time_measure()
        self.logger.debug("testing...")
        for iter_num, (images, labels) in enumerate(train_loader):
            self.logger.debug("iter: {}, load data".format(iter_num))

            # Adjust to CUDA
            images, labels = TorchUtils.to_device(images), TorchUtils.to_device(labels)
            self.logger.debug("iter: {}, data and labels to CUDA:".format(iter_num))

            # Forward-pass of natural images
            self.optimizer.zero_grad()
            if self.adv_learn_alpha != 1:
                outputs, natural_loss = self.__forward_pass(model, images, labels)
                _, predicted = torch.max(outputs.data, 1)
                natural_correct += (predicted == labels).sum().item()
                natural_train_loss_in_ep += natural_loss * len(images)  # natural_loss sum for the epoch
                self.logger.debug("iter: {}, Forward pass".format(iter_num))

            # Back-propagation
            if self.adv_learn_alpha == 0:
                natural_loss.backward()
                self.logger.debug("iter: {}, Backward pass".format(iter_num))
                correct = natural_correct
            else:
                # # The loss is averaged over the minibatch, this doesn't matter at all since the loss magnitude
                # # is only just to determine gradient sign. Each pixel is changed by -+epsilon, no matter
                # # it's gradient magnitude.
                adv_images = attack.create_adversarial_sample(images, labels)
                self.optimizer.zero_grad()
                self.logger.debug("iter: {}, create adversarial data".format(iter_num))

                #### add another sample
                # TODO: move it outside if, otherwise this won't happen for non-adversarial training
                if (iter_num) == 0 and sample_test_data is not None:
                    sample_test_data.requires_grad = False
                    sample_test_true_label.requires_grad = False
                    output_sample_test, loss_sample_test = self.__forward_pass(model, sample_test_data,
                                                                               sample_test_true_label)
                    _, predicted = torch.max(output_sample_test.data, 1)
                    # correct += (predicted == sample_test_true_label).sum().item() #TODO: when calculating accuracy (after epoch) we need to divide by +1
                    if TrainClass.criterion.reduction == 'elementwise_mean':  # Re-average the loss since another image was added
                        loss_sample_test = loss_sample_test / (len(images) + 1)
                        # natural_loss = natural_loss * len(images) / (len(images) + 1) TODO: uncomment
                    loss_sample_test.backward(retain_graph=True)
                    # self.optimizer.zero_grad()
                ####

                outputs, adv_loss = self.__forward_pass(model, adv_images, labels)
                self.logger.debug("iter: {}, Adv. Forward pass".format(iter_num))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total_loss = (1 - self.adv_learn_alpha) * natural_loss + self.adv_learn_alpha * adv_loss #+ loss_sample_test
                total_loss_in_epoch += total_loss * len(images)
                total_loss.backward()
                self.logger.debug("iter: {}, Adv. Backward pass".format(iter_num))

            self.optimizer.step()
            self.logger.debug("iter: {}, Optimizer step".format(iter_num))
        self.scheduler.step()
        natural_train_loss_in_ep /= len(train_loader.dataset)
        total_loss_in_epoch /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        return total_loss_in_epoch, natural_train_loss_in_ep, train_acc

    @classmethod
    def __forward_pass(cls, model, images, labels, loss_func='default'):
        outputs = model(images, labels)
        if loss_func == 'default':
            loss = cls.criterion(outputs, labels)  # Negative log-loss
        else:
            criterion = nn.NLLLoss()
            loss = criterion(torch.log(outputs), labels)
        return outputs, loss

    @classmethod
    def eval_model(cls, model, dataloader, attack = get_attack("no_attack"), loss_func='default'):
        """
        Evaluate the performance of the model on the train/test sets.
        :param model: the model that will be evaluated.
        :param dataloader: trainset or testset on which the evaluation will executed.
        :return: the loss and accuracy on the testset.
        """
        model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for iter_num, (data, labels) in enumerate(dataloader):
                print("eval_model iter_num: {}".format(iter_num))
                data, labels = TorchUtils.to_device(data), TorchUtils.to_device(labels)
                with torch.enable_grad():
                    adv_data = attack.create_adversarial_sample(data, labels)
                    outputs, batch_loss = cls.__forward_pass(model, adv_data, labels, loss_func)
                loss += batch_loss * len(adv_data)  # loss sum for all the batch
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                if (predicted == labels).sum().item() == 1:
                    print("correct prediction in iter_num: {}".format(iter_num))

        acc = correct / len(dataloader.dataset)
        loss /= len(dataloader.dataset)
        return loss, acc


def eval_single_sample(model, test_sample_data):
    """
    Predict the probabilities assignment of the test sample by the model
    :param model: the model which will evaluate the test sample
    :param test_sample_data: the data of the test sample
    :return: prob: probabilities vector. pred: the class prediction of the test sample
    """
    # test_sample = (data, label)

    # Test the sample
    model.eval()

    # if next(model.parameters()).is_cuda:  # This only checks the first iteration from parameters(), a better way is to loop over all parameters
    #     sample_data = test_sample_data.cuda()
    # else:
    #     sample_data = test_sample_data.cpu()
    sample_data = TorchUtils.to_device(test_sample_data)
    if len(sample_data.shape) == 3:
        sample_data = sample_data.unsqueeze(0)  # make the single sample 4-dim tensor

    with torch.no_grad():
        output = model(sample_data).detach().cpu()

        # Prediction
        pred = output.max(1, keepdim=True)[1]
        pred = pred.numpy()[0][0]

        # Extract prob
        prob = F.softmax(output, dim=-1)
        prob = prob.numpy().tolist()[0]
    return prob, pred


def set_bn_eval(model):
    """
    Freeze batch normalization layers for better control on training
    :param model: the model which the freeze of BN layers will be executed
    :return: None, the freeze is in place on the model.
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        model.eval()


def execute_pnml_training(train_params: dict, params_init_training: dict, dataloaders_input: dict,
                          sample_test_data_trans, sample_test_true_label, idx: int,
                          model_base_input, logger, genie_only_training: bool=False, adv_train: bool=False):
    """
    Execute the PNML procedure: for each label train the model and save the prediction afterword.
    :param train_params: parameters of training the model for each label
    :param train_class: the train_class is used to train and eval the model
    :param dataloaders_input: dataloaders which contains the trainset
    :param sample_test_data_trans: the data of the test sample that will be evaluated
    :param sample_test_true_label: the true label of the test sample
    :param idx: the index in the testset dataset of the test sample
    :param model_base_input: the base model from which the train will start
    :param logger: logger class to print logs and save results to file
    :param genie_only_training: calculate only genie probability for speed up when debugging
    :param adv_train: train the test sample with or without adversarial regularizer
    :return: None
    """

    # Check train_params contains all required keys
    required_keys = ['lr', 'momentum', 'step_size', 'gamma', 'weight_decay', 'epochs']
    for key in required_keys:
        if key not in train_params:
            logger.logger.error('The key: %s is not in train_params' % key)
            raise ValueError('The key: %s is not in train_params' % key)

    classes_trained = dataloaders_input['classes']
    if 'classes_cifar100' in dataloaders_input:
        classes_true = dataloaders_input['classes_cifar100']
    elif 'classes_svhn' in dataloaders_input:
        classes_true = dataloaders_input['classes_svhn']
    elif 'classes_noise' in dataloaders_input:
        classes_true = dataloaders_input['classes_noise']
    else:
        classes_true = classes_trained

    # Iteration of all labels
    if genie_only_training:
        trained_label_list = [sample_test_true_label.tolist()]
    else:
        trained_label_list = range(len(classes_trained))

    for trained_label in trained_label_list:
        time_trained_label_start = time.time()

        if adv_train:
            raise NotImplementedError('insert the sample to the train requires to take the sample before transform and attack')
            # # Insert test sample to train dataset and train the test sample with adversarial regularizer
            # dataloaders = deepcopy(dataloaders_input)
            # trainloader_with_sample = insert_sample_to_dataset(dataloaders['train'], sample_test_data, trained_label)
            # dataloaders['train'] = trainloader_with_sample
        else:
            sample_to_insert_label_expand = torch.tensor(np.expand_dims(trained_label, 0), dtype=torch.int64)  # make the label to tensor array type (important for loss calculation)
            if len(sample_test_data_trans.shape) == 3:
                sample_test_data_trans = sample_test_data_trans.unsqueeze(0)# make the single sample 4-dim tensor

        # Execute transformation - for training and evaluating the test sample
        # sample_test_data_for_trans = copy.deepcopy(sample_test_data)
        # if len(sample_test_data.shape) == 2:
        #     print("mek")
        #     sample_test_data_for_trans = sample_test_data_for_trans.unsqueeze(2).numpy()
        # sample_test_data_trans = dataloaders_input['test'].dataset.transform(sample_test_data_for_trans)




        # Train model
        model = deepcopy(model_base_input)
        train_class = TrainClass(filter(lambda p: p.requires_grad, model.parameters()),
                                 train_params['lr'], train_params['momentum'], train_params['step_size'],
                                 train_params['gamma'], train_params['weight_decay'],
                                 logger,
                                 params_init_training["adv_alpha"], params_init_training["epsilon"],
                                 params_init_training["attack_type"], params_init_training["pgd_iter"],
                                 params_init_training["pgd_step"]
                                 )
        train_class.eval_test_during_train = True
        train_class.freeze_batch_norm = True
        # model, train_loss, test_loss = train_class.train_model(model, dataloaders, train_params['epochs'])
        if adv_train:
            model, train_loss, test_loss = train_class.train_model(model, dataloaders_input, train_params['epochs'],
                                                                   sample_test_data=None,
                                                                   sample_test_true_label=None)
        else:
            model, train_loss, test_loss = train_class.train_model(model, dataloaders_input, train_params['epochs'],
                                                               sample_test_data=sample_test_data_trans,
                                                               sample_test_true_label=sample_to_insert_label_expand)
        time_trained_label = time.time() - time_trained_label_start

        # Evaluate with base model
        prob, pred = eval_single_sample(model, sample_test_data_trans)

        # Save to file
        logger.add_entry_to_results_dict(idx, str(trained_label), prob, train_loss, test_loss)
        logger.info(
            'idx=%d trained_label=[%d,%s], true_label=[%d,%s] predict=[%d], loss [train, test]=[%f %f], time=%4.2f[s]'
            % (idx, trained_label, classes_trained[trained_label],
               sample_test_true_label, classes_true[sample_test_true_label],
               np.argmax(prob),
               train_loss, test_loss,
               time_trained_label))

execute_pnml_adv_fix_ind = 0
def execute_pnml_adv_fix(pnml_params: dict, params_init_training: dict, dataloaders_input: dict,
                          sample_test_data_trans, sample_test_true_label, idx: int,
                          model_base_input, logger, genie_only_training: bool=False):
    """
    Execute the PNML procedure: for each label train the model and save the prediction afterword.
    :param pnml_params: parameters of training the model for each label
    :param train_class: the train_class is used to train and eval the model
    :param dataloaders_input: dataloaders which contains the trainset
    :param sample_test_data_trans: the data of the test sample that will be evaluated
    :param sample_test_true_label: the true label of the test sample
    :param idx: the index in the testset dataset of the test sample
    :param model_base_input: the base model from which the train will start
    :param logger: logger class to print logs and save results to file
    :param genie_only_training: calculate only genie probability for speed up when debugging
    :return: None
    """

    # Check pnml_params contains all required keys
    required_keys = ['lr', 'momentum', 'step_size', 'gamma', 'weight_decay', 'epochs']
    for key in required_keys:
        if key not in pnml_params:
            logger.logger.error('The key: %s is not in pnml_params' % key)
            raise ValueError('The key: %s is not in pnml_params' % key)

    classes_trained = dataloaders_input['classes']
    if 'classes_cifar100' in dataloaders_input:
        classes_true = dataloaders_input['classes_cifar100']
    elif 'classes_svhn' in dataloaders_input:
        classes_true = dataloaders_input['classes_svhn']
    elif 'classes_noise' in dataloaders_input:
        classes_true = dataloaders_input['classes_noise']
    else:
        classes_true = classes_trained

    # Iteration of all labels
    if genie_only_training:
        trained_label_list = [sample_test_true_label.tolist()]
    else:
        trained_label_list = range(len(classes_trained))
    model = deepcopy(model_base_input)  # working on a single sample, it is reasonable to assume cpu is better
    refinement = get_attack(pnml_params['fix_type'], model, pnml_params['epsilon'], pnml_params['pgd_iter'],
                           pnml_params['pgd_step'], pnml_params['pgd_rand_start'], get_dataset_min_max_val(dataloaders_input['dataset_name']),
                           pnml_params['pgd_test_restart_num'])
    for fix_to_label in trained_label_list:
        time_trained_label_start = time.time()

        fix_label_expand = TorchUtils.to_device(torch.tensor(np.expand_dims(fix_to_label, 0), dtype=torch.int64))
        true_label_expand = TorchUtils.to_device(torch.tensor(np.expand_dims(sample_test_true_label, 0), dtype=torch.int64))  # make the label to tensor array type (important for loss calculation)
        if len(sample_test_data_trans.shape) == 3:
            sample_test_data_trans = TorchUtils.to_device(sample_test_data_trans.unsqueeze(0))# make the single sample 4-dim tensor

        time_trained_label = time.time() - time_trained_label_start


        # Evaluate with base model
        assert(not model.training)
        x_refine = refinement.create_adversarial_sample(sample_test_data_trans, true_label_expand, fix_label_expand)
        # assert(sample_test_data_trans.grad is None)
        prob, pred = eval_single_sample(model, x_refine)

        global execute_pnml_adv_fix_ind
        if execute_pnml_adv_fix_ind == 0:
            from utilities import plt_img
            # plt_img(x_refine, 0)


        # Save to file
        logger.add_entry_to_results_dict(idx, str(fix_to_label), prob, -1, -1)
        logger.info(
            'idx=%d fix_to_label=[%d,%s], true_label=[%d,%s] predict=[%d], time=%4.2f[s]'
            % (idx, fix_to_label, classes_trained[fix_to_label],
               sample_test_true_label, classes_true[sample_test_true_label],
               np.argmax(prob),
               time_trained_label))
    execute_pnml_adv_fix_ind = execute_pnml_adv_fix_ind + 1



def freeze_model_layers(model, max_freeze_layer: int, logger):
    """
    Freeze model layers until max_freeze_layer, all others can be updated
    :param model: to model on which the freeze will be executed
    :param max_freeze_layer: the maximum depth of freeze layer
    :param logger: logger class in order to print the layers and their status (freeze/unfreeze)
    :return: model with freeze layers
    """
    # todo: currently max_freeze_layer 0 and 1 are the same. move ct+=1 to the end
    ct = 0
    for child in model.children():
        ct += 1
        if ct < max_freeze_layer:
            logger.info('Freeze Layer: idx={}, name={}'.format(ct, child))
            for param in child.parameters():
                param.requires_grad = False
            continue
        logger.info('UnFreeze Layer: idx={}, name={}'.format(ct, child))
    return model


def tensor_to_cuda(tensor: torch.tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor
