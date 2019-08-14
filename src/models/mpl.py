import torch.nn as nn
from torch.nn import functional as F
import torch.autograd
from .model_utils import ModelTemplate


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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x, *args):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

mnist_std = 0.3081
mean_mnist = 0.1307
mnist_min_val = (0 - mean_mnist) / mnist_std
mnist_max_val = (1 - mean_mnist) / mnist_std
class PnmlMnistClassifier(MNISTClassifier):
    def __init__(self, eps=0.3, gamma=0.0):
        super(PnmlMnistClassifier, self).__init__()
        self.eps = eps * (mnist_max_val - mnist_min_val)
        self.gamma = gamma * (mnist_max_val - mnist_min_val)
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # self.fc1 = nn.Linear(1024, 10)
        self.clamp = (mnist_min_val, mnist_max_val)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, true_label):
        # x.require_grad = True
        # x = x.detach()

        # x.requires_grad = True
        # log_prob = self.forward_base_model(x)
        # loss = self.loss_fn(log_prob, true_label)
        # grad = torch.autograd.grad(loss, x, create_graph=True, allow_unused=True)[0]
        # x_adv = x + grad + torch.sign(grad) * self.eps
        # x.requires_grad = False

        x_adv = x
        x_adv.requires_grad = True
        adv_log_prob = self.forward_base_model(x_adv)
        adv_loss = self.loss_fn(adv_log_prob, true_label)
        adv_grad = torch.autograd.grad(adv_loss, x_adv, create_graph=True, allow_unused=True)[0]
        x_genie = (x_adv - torch.sign(adv_grad) * self.gamma) #.clamp(*self.clamp)
        return self.forward_base_model(x_genie)

    def forward_base_model(self, x):
        return super(PnmlMnistClassifier, self).forward(x)


class PnmlMnistClassifier2(ModelTemplate):
    def __init__(self, base_model, eps=0.3):
        super(PnmlMnistClassifier2, self).__init__()
        self.eps = eps * (mnist_max_val - mnist_min_val)
        self.base_model = base_model

    def forward(self, x, true_label):
        # x.require_grad = True
        # x = x.detach()
        x.requires_grad = True
        loss_fn = torch.nn.CrossEntropyLoss()
        log_prob = self.forward_base_model(x)
        loss = loss_fn(log_prob, true_label)
        grad = torch.autograd.grad(loss, x, create_graph=True, allow_unused=True)[0]
        x_adv = x + grad + torch.sign(grad) * self.eps
        x.requires_grad = False
        return self.forward_base_model(x_adv)

    def forward_base_model(self, x):
        return self.base_model(x)


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

