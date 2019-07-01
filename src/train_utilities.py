import copy
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
from dataset_utilities import insert_sample_to_dataset, mnist_min_val, mnist_max_val


class TrainClass:
    """
    Class which execute train on a DNN model.
    """
    criterion = nn.CrossEntropyLoss()
    def __init__(self, params_to_train, learning_rate: float, momentum: float, step_size: list, gamma: float,
                 weight_decay: float, logger=None, adv_learn_alpha=0, adv_learn_eps=0.05, attack_type: str = 'pgd',
                 pgd_iter: int = 30, pgd_step: float = 0.01, pgd_random: bool = True):
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

        self.num_epochs = 20
        self.logger = logger if logger is not None else logging.StreamHandler(sys.stdout)
        self.eval_test_during_train = True
        self.eval_test_in_end = True
        self.print_during_train = True

        # Optimizer
        self.optimizer = optim.SGD(params_to_train,
                                   lr=learning_rate,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=step_size,
                                                        gamma=gamma)
        self.freeze_batch_norm = True
        self.adv_learn_alpha = adv_learn_alpha
        self.adv_learn_eps = adv_learn_eps
        self.attack_type = attack_type
        self.pgd_iter = pgd_iter
        self.pgd_step = pgd_step
        self.pgd_random = pgd_random

    def train_model(self, model, dataloaders, num_epochs: int = 10, acc_goal=None,
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
        print("Use GPU:" + str(torch.cuda.is_available()))
        model = model.cuda() if torch.cuda.is_available() else model
        attack = get_attack(self.attack_type, model, self.adv_learn_eps, self.pgd_iter, self.pgd_step,
                            self.pgd_random, (mnist_min_val, mnist_max_val))

        self.num_epochs = num_epochs
        train_loss, train_acc = torch.tensor([-1.]), torch.tensor([-1.])
        epoch_time = 0
        lr = 0

        # Loop on epochs
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()
            total_loss_in_epoch, train_loss, train_acc = self.__train(model, dataloaders['train'], attack,
                                                                      sample_test_data, sample_test_true_label)
            if self.eval_test_during_train is True:
                assert(dataloaders.__contains__('test'))
                test_loss, test_acc = self.eval(model, dataloaders['test'])
            else:
                test_loss, test_acc = torch.tensor([-1.]), torch.tensor([-1.])
            epoch_time = time.time() - epoch_start_time

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            self.logger.info('[%d/%d] [train test] loss =[%f %f] adv_loss=[%f], acc=[%f %f], lr=%f, epoch_time=%.2f'
                             % (epoch, self.num_epochs - 1,
                                train_loss, test_loss, total_loss_in_epoch, train_acc, test_acc,
                                lr, epoch_time))

            # Stop training if desired goal is achieved
            if acc_goal is not None and train_acc >= acc_goal:
                break

        if 'test' in dataloaders:
            test_loss, test_acc = self.eval(model, dataloaders['test'])
        else:
            test_loss, test_acc = torch.tensor([-1.]), torch.tensor([-1.])

        train_loss_output = float(train_loss.cpu().detach().numpy().round(16))
        test_loss_output = float(test_loss.cpu().detach().numpy().round(16))
        # Print and save
        self.logger.info('----- [train test] loss =[%f %f], adv_loss=[%f], acc=[%f %f] epoch_time=%.2f' %
                         (train_loss, test_loss, total_loss_in_epoch, train_acc, test_acc,
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
        train_loss = 0
        correct = 0
        loss_sample_test = 0
        # max_iter = np.ceil(len(train_loader.dataset) / train_loader.batch_size) #TODO: from some reason I am missing the last batch but the dataloader should not drop the last batch
        if sample_test_data is not None:

            sample_test_data = sample_test_data.cuda() if torch.cuda.is_available() else sample_test_data
            sample_test_true_label = sample_test_true_label.cuda() if torch.cuda.is_available() else sample_test_true_label
            sample_test_data.requires_grad = False
            sample_test_true_label.requires_grad = False

        # Iterate over dataloaders
        for iter_num, (images, labels) in enumerate(train_loader):

            # Adjust to CUDA
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            # Forward
            self.optimizer.zero_grad()
            outputs, loss = self.__forward_pass(model, images, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            train_loss += loss * len(images)  # loss sum for the epoch

            # Back-propagation

            if self.adv_learn_alpha is 0:
                loss.backward()
            else:
                # # The loss is averaged over the minibatch, this doesn't matter at all since the loss magnitude
                # # is only just to determine gradient sign. Each pixel is changed by -+epsilon, no matter
                # # it's gradient magnitude.
                adv_images = attack.create_adversarial_sample(images, labels)
                self.optimizer.zero_grad()

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
                        # loss = loss * len(images) / (len(images) + 1) TODO: uncomment
                    loss_sample_test.backward(retain_graph=True)
                    # self.optimizer.zero_grad()
                ####


                _, adv_loss = self.__forward_pass(model, adv_images, labels)
                # outputs = model(adv_images)
                # adv_loss = TrainClass.criterion(outputs, labels)  # Negative log-loss
                total_loss = (1 - self.adv_learn_alpha) * loss + self.adv_learn_alpha * adv_loss #+ loss_sample_test
                total_loss_in_epoch += total_loss * len(images)
                total_loss.backward()

            self.optimizer.step()

        self.scheduler.step()
        train_loss /= len(train_loader.dataset)
        total_loss_in_epoch /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        return total_loss_in_epoch, train_loss, train_acc

    @classmethod
    def __forward_pass(cls, model, images, labels):
        outputs = model(images)
        loss = cls.criterion(outputs, labels)  # Negative log-loss
        return outputs, loss

    @classmethod
    def eval(cls, model, dataloader):
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
            for data, labels in dataloader:
                data = data.cuda() if torch.cuda.is_available() else data
                labels = labels.cuda() if torch.cuda.is_available() else labels

                outputs, batch_loss = cls.__forward_pass(model, data, labels)
                loss += batch_loss * len(data)  # loss sum for all the batch
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

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

    if next(model.parameters()).is_cuda:  # This only checks the first iteration from parameters(), a better way is to loop over all parameters
        sample_data = test_sample_data.cuda()
    else:
        sample_data = test_sample_data.cpu()
    # sample_data = test_sample_data.cuda() if torch.cuda.is_available() else test_sample_data
    if len(sample_data.shape) == 3:
        sample_data = sample_data.unsqueeze(0)  # make the single sample 4-dim tensor
    output = model(sample_data)

    # Prediction
    pred = output.max(1, keepdim=True)[1]
    pred = pred.cpu().detach().numpy().round(16)[0][0]

    # Extract prob
    prob = F.softmax(output, dim=-1)
    prob = prob.cpu().detach().numpy().round(16).tolist()[0]
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
                                 logger.logger,
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
    model = deepcopy(model_base_input.to("cpu"))  # working on a single sample, it is reasonable to assume cpu is better
    refinement = get_attack(pnml_params['fix_type'], model, pnml_params['epsilon'], pnml_params['pgd_iter'],
                           pnml_params['pgd_step'], pnml_params['pgd_rand_start'], (mnist_min_val, mnist_max_val),
                           pnml_params['pgd_test_restart_num'])
    for fix_to_label in trained_label_list:
        time_trained_label_start = time.time()

        fix_label_expand = torch.tensor(np.expand_dims(fix_to_label, 0), dtype=torch.int64)
        true_label_expand = torch.tensor(np.expand_dims(sample_test_true_label, 0), dtype=torch.int64)  # make the label to tensor array type (important for loss calculation)
        if len(sample_test_data_trans.shape) == 3:
            sample_test_data_trans = sample_test_data_trans.unsqueeze(0)# make the single sample 4-dim tensor

        time_trained_label = time.time() - time_trained_label_start


        # Evaluate with base model
        x_refine = refinement.create_adversarial_sample(sample_test_data_trans, true_label_expand, fix_label_expand)
        prob, pred = eval_single_sample(model, x_refine)

        global execute_pnml_adv_fix_ind
        if execute_pnml_adv_fix_ind == 0:
            from test_net_script import plt_img
            plt_img(x_refine, 0)


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
