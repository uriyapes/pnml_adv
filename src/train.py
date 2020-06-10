import jsonargparse
import os
import torch
from torch.nn import functional as F
import numpy as np
import logger_utilities
from experimnet_utilities import Experiment
from utilities import TorchUtils
TorchUtils.set_rnd_seed(1)
# Uncomment for performance. Comment for debug and reproducibility
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
from train_utilities import TrainClass
from eval import eval_adversarial_dataset, eval_pnml_blackbox, eval_all


def display_decision_planes(model, dataloader):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    inputs = torch.Tensor(pos)
    output = model(inputs)
    prob = F.softmax(output, dim=2).detach().numpy()[:,:,0]

    fig = plt.figure(0)
    ax = plt.gca()
    cs = ax.contourf(X, Y, prob, 10)


    CS2 = ax.contour(cs, levels=cs.levels[::1], colors='r', origin='lower')
    cbar = fig.colorbar(cs)
    cbar.add_lines(CS2)
    plt.clabel(CS2, inline=1, fontsize=10)
    # labels = ['line1', 'line2','line3','line4',
    #            'line5', 'line6']
    # for i in range(len(labels)):
    #     CS2.collections[i].set_label(labels[i])
    # plt.legend(loc='upper right')

    # Display samples
    for iter_num, (samples, labels) in enumerate(dataloader):
        x = samples.numpy()
        x_0 = x[labels == 0,:]
        x_1 = x[labels == 1, :]
        plt.scatter(x_0[:,0], x_0[:,1], c='r')
        plt.scatter(x_1[:, 0], x_1[:, 1], c='b')
    plt.show()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description='General arguments', default_meta=False)
    parser.add_argument('-t', '--general.experiment_type', default='synthetic',
                        help='Type of experiment to execute', type=str)
    parser.add_argument('-p', '--general.param_file_path', default=os.path.join('./src/parameters', 'train_synthetic_params.json'),
                        help='param file path used to load the parameters file containing default values to all '
                             'parameters', type=str)
    # parser.add_argument('-p', '--general.param_file_path', default='src/tests/test_mnist_pgd_with_pnml_expected_result/params.json',
    #                     help='param file path used to load the parameters file containing default values to all '
    #                          'parameters', type=str)
    parser.add_argument('-o', '--general.output_root', default='output', help='the output directory where results will be saved', type=str)
    args = jsonargparse.namespace_to_dict(parser.parse_args())
    general_args = args.pop('general')

    exp = Experiment(general_args, args)
    params = exp.get_params()
    logger_utilities.init_logger(logger_name=exp.get_exp_name(), output_root=exp.output_dir)
    logger = logger_utilities.get_logger()

    model_to_train = exp.get_model(exp.params['model']['model_arch'], exp.params['model']['ckpt_path'])
    dataloaders = exp.get_dataloaders()

    logger.info('Execute basic training')
    params_init_training = params['initial_training']
    train_class = TrainClass(filter(lambda p: p.requires_grad, model_to_train.parameters()),
                             params_init_training['lr'],
                             params_init_training['momentum'],
                             params_init_training['step_size'],
                             params_init_training['gamma'],
                             params_init_training['weight_decay'],
                             logger,
                             params_init_training["adv_attack_train"], params_init_training["adv_alpha"],
                             params_init_training["save_model_every_n_epoch"])
    train_class.eval_test_during_train = True if params_init_training['eval_test_every_n_epoch'] is not None else False
    train_class.freeze_batch_norm = False
    acc_goal = params_init_training['acc_goal'] if 'acc_goal' in params_init_training else None
    model_trained, train_loss, test_loss = train_class.train_model(model_to_train, dataloaders,
                                                                params_init_training['epochs'], acc_goal,
                                                                params_init_training['eval_test_every_n_epoch'])
    torch.save(model_trained.state_dict(),
               os.path.join(logger.output_folder, '%s_model_%f.pt' % (exp.get_exp_name(), train_loss)))
    logger.info('Done basic training')

    # display_decision_planes(model_trained, dataloaders['test'])
    attack = exp.get_attack_for_model(model_trained)

    logger.info('Evaluate natural accuracy and robustness for base model and Adversarial pNML scheme:')
    adv, adv_pnml, natural, natural_pnml = eval_all(model_trained, dataloaders['test'], attack, exp)
    logger.info("Base model adversarial - Accuracy: {}, Loss: {}".format(adv.get_accuracy(), adv.get_mean_loss()))
    logger.info("Pnml model adversarial - Accuracy: {}, Loss: {}".format(adv_pnml.get_accuracy(), adv_pnml.get_mean_loss()))
    logger.info("Base model natural - Accuracy: {}, Loss: {}".format(natural.get_accuracy(), natural.get_mean_loss()))
    logger.info("Pnml model natural - Accuracy: {}, Loss: {}".format(natural_pnml.get_accuracy(), natural_pnml.get_mean_loss()))
    #
    # adv_pnml = eval_pnml_blackbox(model_trained, adv, exp)
    # loss = adv_pnml.get_mean_loss()
    # acc = adv_pnml.get_accuracy()
    # logger.info("Pnml model - Accuracy: {}, Loss: {}".format(acc, loss))


