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


class PnmlModel(ModelTemplate):
    def __init__(self, base_model, params, clamp, num_classes=10):
        super().__init__()
        self.params = params
        self.gamma = params['epsilon']
        self.clamp = clamp
        self.base_model = base_model
        self.refine = get_attack("fgsm", self.base_model, self.gamma, params["pgd_iter"], params["pgd_step"], False,
                                 self.clamp, 1)
        # self.refine = get_attack("pgd", self.base_model, self.gamma, params["pgd_iter"], params["pgd_step"], False,
        #                          self.clamp, 1)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')  # reduction='none' The loss is config with reduction='none' so the grad of each sample won't be effected by other samples
        #     TODO : support larger batch sizes.
        self.regularization = torch.zeros([1])  # This value stores the risk
        # self.training = False
        self.num_classes = num_classes
        self.genie_prob = None
        self.pnml_model = True

    def forward(self, x):
        genie_prob = torch.zeros([x.shape[0], self.num_classes], requires_grad=True).to(x.device)
        for label in range(self.num_classes):
            # print("Label: {}".format(label))
            torch_label = torch.ones([x.shape[0]], dtype=torch.long).to(x.device) * label
            genie_prob[:, label] = F.softmax(self.forward_genie(x, torch_label), dim=1)[:, label]
        self.genie_prob = genie_prob
        pnml_prob = genie_prob / genie_prob.sum(dim=1, keepdim=True)
        pnml_prob_sum = genie_prob.sum(dim=1, keepdim=False)
        self.regularization = torch.log(pnml_prob_sum)  # This is the regret, each sample in the batch has it's own regret.
        # assert(torch.allclose(pnml_prob.sum(dim=1), ))
        return pnml_prob

    def forward_genie(self, x, label):
        # x.requires_grad = True
        x_genie = self.refine.create_adversarial_sample(x, None, label) #.detach()  # TODO: remove detach  and no_grad_flag=True to enable white-box attacks
        return self.forward_base_model(x_genie, no_grad_flag=False) #.detach()

    def get_genie_prob(self):
        return self.genie_prob.clone()

    def calc_log_prob(self, x):
        return torch.log(self.__call__(x))

    def forward_genie_sign_approx(self, x, label):
        x.requires_grad = True
        output = self.forward_base_model(x)
        adv_loss = self.loss_fn(output, label)
        adv_grad = torch.autograd.grad(adv_loss, x, create_graph=True, allow_unused=True)[0] #TODO: remove create_graph
        adv_grad_sign = adv_grad * 1500
        x_genie = (x - adv_grad_sign * self.gamma * (mnist_max_val - mnist_min_val))
        return self.forward_base_model(x_genie)

    def forward_base_model(self, x, no_grad_flag=False):
        if no_grad_flag:
            with torch.no_grad():
                return self.base_model(x)
        else:
            return self.base_model(x)

    def state_dict(self):
        return self.base_model.state_dict()


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

