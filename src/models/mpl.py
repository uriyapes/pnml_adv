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
        output = self.fc1(x)
        return output
        # return F.log_softmax(x, dim=1)



mnist_std = 0.3081
mean_mnist = 0.1307
mnist_min_val = (0 - mean_mnist) / mnist_std
mnist_max_val = (1 - mean_mnist) / mnist_std
class PnmlMnistClassifier(MNISTClassifier):
    def __init__(self, eps=0.3, gamma=0.1):
        super(PnmlMnistClassifier, self).__init__()
        self.eps = eps * (mnist_max_val - mnist_min_val)
        self.gamma = gamma * (mnist_max_val - mnist_min_val)
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # self.fc1 = nn.Linear(1024, 10)
        self.clamp = (mnist_min_val, mnist_max_val)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, true_label):
        # x.require_grad = True
        # x = x.detach()

        # x.requires_grad = True
        # log_prob = self.forward_base_model(x)
        # loss = self.loss_fn(log_prob, true_label)
        # grad = torch.autograd.grad(loss, x, create_graph=True, allow_unused=True)[0]
        # x_adv = x + grad + torch.sign(grad) * self.eps
        # x.requires_grad = False
        num_classes = 10
        genie_prob = torch.zeros([x.shape[0], num_classes], requires_grad=True).to(x.device)
        for label in range(num_classes):
            torch_label = torch.ones([x.shape[0]], dtype=torch.long).to(x.device) * label
            genie_prob[:, label] = F.softmax(self.forward_genie(x, torch_label), dim=1)[:, label]
        pnml_prob = genie_prob / genie_prob.sum(dim =1, keepdim=True)
        # assert(torch.allclose(pnml_prob.sum(dim=1), ))
        return pnml_prob

    def forward_genie(self, x, label):
        # x = x.clone()
        x.requires_grad = True
        # x_adv.requires_grad = True
        output = self.forward_base_model(x)
        adv_loss = self.loss_fn(output, label) #TODO : make sure the loss doesn't do mean() so grad value will be the same no matter the batch size.
        adv_grad = torch.autograd.grad(adv_loss, x, create_graph=True, allow_unused=True)[0]
        # adv_grad_sign = torch.sign(adv_grad)
        adv_grad_sign = adv_grad * 500
        # x_genie = (x - adv_grad_sign * self.gamma)
        x_genie = (x - adv_grad_sign * self.gamma)#.clamp(*self.clamp)
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

