import torch.nn as nn
from torch.nn import functional as F
import torch.autograd
from .model_utils import ModelTemplate
from adversarial.attacks import get_refiner
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
    def __init__(self, base_model, params, clamp, num_classes=10, pnml_model_keep_grad=True):
        super().__init__()
        self.params = params
        self.gamma = params['epsilon']
        self.clamp = clamp
        self.base_model = base_model
        self.pnml_model_keep_grad = pnml_model_keep_grad
        self.refine = get_refiner(params, self.base_model, self.clamp, num_class=num_classes)
        # self.refine = get_attack("pgd", self.base_model, self.gamma, params["pgd_iter"], params["pgd_step"], False,
        #                          self.clamp, 1)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')  # reduction='none' The loss is config with reduction='none' so the grad of each sample won't be effected by other samples
        self.regularization = torch.zeros([1])  # This value stores the risk
        # self.training = False
        self.num_classes = num_classes
        self.genie_prob = None
        self.pnml_model = True

    # def forward(self, x):
    #     genie_prob = torch.zeros([x.shape[0], self.num_classes], requires_grad=True).to(x.device)
    #     for label in range(self.num_classes):
    #         # print("Label: {}".format(label))
    #         torch_label = torch.ones([x.shape[0]], dtype=torch.long).to(x.device) * label
    #         genie_prob[:, label] = F.softmax(self.forward_genie(x, torch_label), dim=1)[:, label]
    #     self.genie_prob = genie_prob
    #     pnml_prob = genie_prob / genie_prob.sum(dim=1, keepdim=True)
    #     pnml_prob_sum = genie_prob.sum(dim=1, keepdim=False)
    #     self.regularization = torch.log(pnml_prob_sum)  # This is the regret, each sample in the batch has it's own regret.
    #     # assert(torch.allclose(pnml_prob.sum(dim=1), ))
    #     return pnml_prob

    def forward(self, x, get_logits=False):
        batch_size = x.shape[0]
        genie_prob = torch.zeros([batch_size, self.num_classes], requires_grad=False, device=x.device)
        x_refined = self.refine.create_refined_samples(x)  #TODO: if pnml_model_keep_grad=False, detach the refined samples inside the create_adversarial_sample method
        x_refined_flat = torch.flatten(x_refined, start_dim=0, end_dim=1)
        with torch.set_grad_enabled(self.pnml_model_keep_grad):
            prob_x_refine = self.forward_base_model(x_refined_flat)
            # if get_logits is False:
            prob_x_refine = F.softmax(prob_x_refine, dim=1)
            prob_x_refine = prob_x_refine.view(self.num_classes, batch_size, self.num_classes)
            for label in range(self.num_classes):
                # print("Label: {}".format(label))
                genie_prob[:, label] = prob_x_refine[label, :, label]  # Take the probability of the refined label
            if get_logits:
                self.genie_prob = None
                return genie_prob
            pnml_prob = genie_prob / genie_prob.sum(dim=1, keepdim=True)
            # pnml_prob_sum = genie_prob.sum(dim=1, keepdim=False)
            # self.regularization = torch.log(pnml_prob_sum)  # This is the regret, each sample in the batch has it's own regret.  TODO: add contidion to detach it
            self.genie_prob = genie_prob.detach()
        return pnml_prob

    def forward_genie(self, x, label):
        # x.requires_grad = True
        x_genie = self.refine.create_adversarial_sample(x, None, label)  #.detach()  # TODO: remove detach  and no_grad_flag=True to enable white-box attacks
        return self.forward_base_model(x_genie, True)  #.detach()

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

    def forward_base_model(self, x):
        return self.base_model(x)

    def state_dict(self):
        return self.base_model.state_dict()

    def calc_logits(self, data):
        return self.__call__(data, get_logits=True)


    def eval_batch(self, data, labels, enable_grad: bool = True, loss_type='logit_diff'):
        """
        :param data: the data to evaluate
        :param labels: the labels of the data
        :param enable_grad: Should grad be enabled for later differentiation
        :param model_output_type: "logits" if model output logits or "prob"
        :param loss_type: 'nll' or 'logit_diff'
        :return: batch loss, probability and label prediction.
        """
        self.eval()

        if loss_type == 'nll':
            loss_func = torch.nn.NLLLoss(reduction='none')
            prob = self.__call__(data)
            loss = loss_func(torch.log(prob), labels)
            genie_prob = self.get_genie_prob()
        elif loss_type == 'logit_diff':
            genie_prob = self.__call__(data, get_logits=True)
            label_logits = torch.zeros(data.shape[0], device=TorchUtils.get_device())
            idx = torch.arange(data.shape[0], device=TorchUtils.get_device())
            label_logits[:] = genie_prob[idx, labels]
            genie_prob[idx, labels] = -1 * float('inf')
            max_wrong_logit = genie_prob.max(dim=1)[0]
            genie_prob[idx, labels] = label_logits[:]
            loss = -1*(label_logits - max_wrong_logit)
            prob = genie_prob / genie_prob.sum(dim=1, keepdim=True)

        return loss, prob.detach(), genie_prob.detach()


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


class NetSynthetic(ModelTemplate):
    def __init__(self, input_size=2, hidden_size1=10, hidden_size2=10, hidden_size3=10, hidden_size4=10, num_classes=2):
        super(NetSynthetic, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(hidden_size4, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.relu4(out)

        out = self.fc5(out)
        return out