import torch.nn as nn
from torch.nn import functional as F
import torch.autograd
from .model_utils import ModelTemplate
from adversarial.attacks import get_attack
from utilities import TorchUtils


class Net(ModelTemplate):
    def __init__(self, input_size=28, hidden_size=10, num_classes=10):
        super(Net, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size * input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size ** 2)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MNISTClassifier(ModelTemplate):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.regularization = TorchUtils.to_device(torch.zeros([1]))
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x, *args):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        output = self.fc1(x)
        return output
        # return F.log_softmax(x, dim=1)


mnist_std = 0.3081
mean_mnist = 0.1307
mnist_min_val = (0 - mean_mnist) / mnist_std
mnist_max_val = (1 - mean_mnist) / mnist_std


class PnmlMnistClassifier(ModelTemplate):
    def __init__(self, base_model, params):
        super().__init__()
        self.gamma = params['epsilon'] * (mnist_max_val - mnist_min_val)
        self.clamp = (mnist_min_val, mnist_max_val)
        self.base_model = base_model
        self.refine = get_attack("pgd", self.base_model, self.gamma, params["pgd_iter"], params["pgd_step"], False,
                                 self.clamp, 1)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')  # reduction='none' The loss is config with reduction='none' so the grad of each sample won't be effected by other samples
        #     TODO : support larger batch sizes.
        self.regularization = torch.zeros([1])  # This value stores the risk

    def forward(self, x, true_label):
        num_classes = 10
        genie_prob = torch.zeros([x.shape[0], num_classes], requires_grad=True).to(x.device)
        for label in range(num_classes):
            torch_label = torch.ones([x.shape[0]], dtype=torch.long).to(x.device) * label
            genie_prob[:, label] = F.softmax(self.forward_genie(x, torch_label), dim=1)[:, label]
        pnml_prob = genie_prob / genie_prob.sum(dim=1, keepdim=True)
        pnml_prob_sum = genie_prob.sum(dim=1, keepdim=False)
        self.regularization = torch.log(pnml_prob_sum)  # This is the regret, each sample in the batch has it's own regret.
        # assert(torch.allclose(pnml_prob.sum(dim=1), ))
        return pnml_prob

    def forward_genie(self, x, label):
        # x.requires_grad = True
        x_genie = self.refine.create_adversarial_sample(x, None, label)
        return self.forward_base_model(x_genie)

    def forward_genie_sign_approx(self, x, label):
        x.requires_grad = True
        output = self.forward_base_model(x)
        adv_loss = self.loss_fn(output, label)
        adv_grad = torch.autograd.grad(adv_loss, x, create_graph=True, allow_unused=True)[0] #TODO: remove create_graph
        adv_grad_sign = adv_grad * 1500
        x_genie = (x - adv_grad_sign * self.gamma)
        return self.forward_base_model(x_genie)

    def forward_base_model(self, x):
        return self.base_model(x)


class PnmlMnistClassifier2(MNISTClassifier):
    def __init__(self, params, gamma=0.1):
        super().__init__()
        self.gamma = params['epsilon'] * (mnist_max_val - mnist_min_val)
        self.clamp = (mnist_min_val, mnist_max_val)
        self.base_model = MNISTClassifier()
        self.refine = get_attack('pgd', self.base_model, self.gamma, params["pgd_iter"], params["pgd_step"], params["pgd_rand_start"],
                                 self.clamp, params['pgd_test_restart_num'])
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')  # reduction='none' The loss is config with reduction='none' so the grad of each sample won't be effected by other samples
        #     TODO : support larger batch sizes.

    def forward(self, x, true_label):
        num_classes = 10
        genie_prob = torch.zeros([x.shape[0], num_classes], requires_grad=True).to(x.device)
        for label in range(num_classes):
            torch_label = torch.ones([x.shape[0]], dtype=torch.long).to(x.device) * label
            genie_prob[:, label] = F.softmax(self.forward_genie(x, torch_label), dim=1)[:, label]
        pnml_prob = genie_prob / genie_prob.sum(dim=1, keepdim=True)
        # assert(torch.allclose(pnml_prob.sum(dim=1), ))
        return pnml_prob

    def forward_genie(self, x, label):
        x.requires_grad = True
        x_genie = self.refine.create_adversarial_sample(x, None, label)
        return self.forward_base_model(x_genie)

    def forward_genie_sign_approx(self, x, label):
        x.requires_grad = True
        output = self.forward_base_model(x)
        adv_loss = self.loss_fn(output, label)
        adv_grad = torch.autograd.grad(adv_loss, x, create_graph=True, allow_unused=True)[0] #TODO: remove create_graph
        adv_grad_sign = adv_grad * 1500
        x_genie = (x - adv_grad_sign * self.gamma)
        return self.forward_base_model(x_genie)

    def forward_base_model(self, x):
        return super().forward(x)


class Net_800_400_100(ModelTemplate):
    def __init__(self, input_size=28, hidden_size1=800, hidden_size2=400, hidden_size3=100, num_classes=10):
        super(Net_800_400_100, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size * input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size ** 2)
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        return out

