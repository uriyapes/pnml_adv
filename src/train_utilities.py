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

from dataset_utilities import insert_sample_to_dataset


class TrainClass:
    """
    Class which execute train on a DNN model.
    """

    def __init__(self, params_to_train, learning_rate: float, momentum: float, step_size: list, gamma: float,
                 weight_decay: float, logger=None, adv_learn_eps=0):
        """
        Initialize train class object.
        :param params_to_train: the parameters of pytorch Module that will be trained.
        :param learning_rate: initial learning rate for the optimizer.
        :param momentum:  initial momentum rate for the optimizer.
        :param step_size: reducing the learning rate by gamma each step_size.
        :param gamma:  reducing the learning rate by gamma each step_size.
        :param weight_decay: L2 regularization.
        :param logger: logger class in order to print logs and save results.
        :param adv_learn_eps: The weight that should be given to the adversarial learning regulaizer (0 means none).
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
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=step_size,
                                                        gamma=gamma)
        self.freeze_batch_norm = True
        self.adv_learn_eps = adv_learn_eps

    def train_model(self, model, dataloaders, num_epochs: int = 10, acc_goal=None):
        """
        Train DNN model using some trainset.
        :param model: the model which will be trained.
        :param dataloaders: contains the trainset for training and testset for evaluation.
        :param num_epochs: number of epochs to train the model.
        :param acc_goal: stop training when getting to this accuracy rate on the trainset.
        :return: trained model (also the training of the models happen inplace)
                 and the loss of the trainset and testset.
        """
        model = model.cuda() if torch.cuda.is_available() else model
        self.num_epochs = num_epochs
        train_loss, train_acc = torch.tensor([-1.]), torch.tensor([-1.])
        epoch_time = 0
        lr = 0

        # Loop on epochs
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()
            train_loss, train_acc = self.train(model, dataloaders['train'])
            if self.eval_test_during_train is True:
                test_loss, test_acc = self.test(model, dataloaders['test'])
            else:
                test_loss, test_acc = torch.tensor([-1.]), torch.tensor([-1.])
            epoch_time = time.time() - epoch_start_time

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            self.logger.info('[%d/%d] [train test] loss =[%f %f], acc=[%f %f], lr=%f, epoch_time=%.2f'
                             % (epoch, self.num_epochs - 1,
                                train_loss, test_loss, train_acc, test_acc,
                                lr, epoch_time))

            # Stop training if desired goal is achieved
            if acc_goal is not None and train_acc >= acc_goal:
                break
        test_loss, test_acc = self.test(model, dataloaders['test'])

        # Print and save
        self.logger.info('----- [train test] loss =[%f %f], acc=[%f %f] epoch_time=%.2f' %
                         (train_loss, test_loss, train_acc, test_acc,
                          epoch_time))
        train_loss_output = float(train_loss.cpu().detach().numpy().round(16))
        test_loss_output = float(test_loss.cpu().detach().numpy().round(16))
        return model, train_loss_output, test_loss_output

    def train(self, model, train_loader):
        """
        Execute one epoch of training
        :param model: the model that will be trained.
        :param train_loader: contains the trainset to train.
        :return: the loss and accuracy of the trainset.
        """
        model.train()

        # Turn off batch normalization update
        if self.freeze_batch_norm is True:
            model = model.apply(set_bn_eval)

        train_loss = 0
        correct = 0
        # Iterate over dataloaders
        for iter_num, (images, labels) in enumerate(train_loader):
            # Adjust to CUDA
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels
            if self.adv_learn_eps is not 0:
                images.requires_grad = True
                # adv_images = images.clone()
                # adv_images.requires_grad = True
                # y = y.cuda() if torch.cuda.is_available() else y
                # x = images.detach()
                # x = x.cuda() if torch.cuda.is_available() else x
                # x.requires_grad = True
                # xx = torch.autograd.Variable(images.data.clone(), requires_grad=True)

            # Forward
            self.optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, labels)  # Negative log-loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            train_loss += loss * len(images)  # loss sum for the epoch

            # Back-propagation

            if self.adv_learn_eps is 0:
                loss.backward()
            else:
                # The loss is averaged over the minibatch, this doesn't matter at all since the loss magnitude
                # is only just to determine gradient sign. Each pixel is changed by -+epsilon, no matter
                # it's gradient magnitude.
                loss.backward(retain_graph=True)
                img_grad = images.grad.data
                img_grad_eps = img_grad.sign() * 0.05 # TODO: change to eps
                adv_images = images.data + img_grad_eps
                torch.clamp(adv_images, 0, 1)
                self.optimizer.zero_grad()
                outputs = model(adv_images)
                adv_loss = self.criterion(outputs, labels)  # Negative log-loss
                total_loss = (1 - self.adv_learn_eps) * loss + self.adv_learn_eps * adv_loss
                total_loss.backward()

            self.optimizer.step()

        self.scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        return train_loss, train_acc

    def test(self, model, test_loader):
        """
        Evaluate the performance of the model on the trainset.
        :param model: the model that will be evaluated.
        :param test_loader: testset on which the evaluation will executed.
        :return: the loss and accuracy on the testset.
        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.cuda() if torch.cuda.is_available() else data
                labels = labels.cuda() if torch.cuda.is_available() else labels

                outputs = model(data)
                loss = self.criterion(outputs, labels)
                test_loss += loss * len(data)  # loss sum for all the batch
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        test_acc = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        return test_loss, test_acc


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
    sample_data = test_sample_data.cuda() if torch.cuda.is_available() else test_sample_data
    output = model(sample_data.unsqueeze(0))

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


def execute_pnml_training(train_params: dict, dataloaders_input: dict,
                          sample_test_data, sample_test_true_label, idx: int,
                          model_base_input, logger):
    """
    Execute the PNML procedure: for each label train the model and save the prediction afterword.
    :param train_params: parameters of training the model for each label
    :param dataloaders_input: dataloaders which contains the trainset
    :param sample_test_data: the data of the test sample that will be evaluated
    :param sample_test_true_label: the true label of the test sample
    :param idx: the index in the testset dataset of the test sample
    :param model_base_input: the base model from which the train will start
    :param logger: logger class to print logs and save results to file
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
    for trained_label in range(len(classes_trained)):
        time_trained_label_start = time.time()

        # Insert test sample to train dataset
        dataloaders = deepcopy(dataloaders_input)
        trainloader_with_sample = insert_sample_to_dataset(dataloaders['train'], sample_test_data, trained_label)
        dataloaders['train'] = trainloader_with_sample

        # Train model
        model = deepcopy(model_base_input)
        train_class = TrainClass(filter(lambda p: p.requires_grad, model.parameters()),
                                 train_params['lr'], train_params['momentum'], train_params['step_size'],
                                 train_params['gamma'], train_params['weight_decay'],
                                 logger.logger)
        train_class.eval_test_during_train = False
        train_class.freeze_batch_norm = True
        model, train_loss, test_loss = train_class.train_model(model, dataloaders, train_params['epochs'])
        time_trained_label = time.time() - time_trained_label_start

        # Execute transformation
        sample_test_data_for_trans = copy.deepcopy(sample_test_data)
        if len(sample_test_data.shape) == 2:
            sample_test_data_for_trans = sample_test_data_for_trans.unsqueeze(2).numpy()
        sample_test_data_trans = dataloaders['test'].dataset.transform(sample_test_data_for_trans)

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
