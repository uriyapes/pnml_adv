# Parameters description
Each json file contains the parameters for a different dataset experiment (toy dataset, MNIST, CIFAR10, ImageNet).
The parameters are divided to 4 subgroups:
1. General parameters
2. adv_attack_test - Set the parameters for the adversarial attack 
3. model - Configure the model and whether to use adversarial pNML.
4. fit_to_sample - adversarial pNML refinement parameters.
5. initial_training - Configure the training.


The following sections detail the effect of each parameter:
## General parameters
```
    "exp_type": The experiment dataset: 'mnist_adversarial', 'cifar_adversarial', 'imagenet_adversarial', 'synthetic'
    "batch_size": Number of samples in a batch
    "num_workers": Number of workers for dataset
    "freeze_layer": deprecated
    "num_classes": Adjust the model to classify only num_classes. For ImageNet set to 100, for synthetic 2, else 10.
```
## adv_attack_test parameters
```
"white_box": set true for white-box attack (PGD for base model or adaptive attack for adversarial pNML scheme). Set false to use transfer attack (PGD attack against adversarial pNML scheme)
"black_box_adv_path": Path to adversarials.t object that contains the adversarial samples that will be tested.
"attack_type": "pgd"
"epsilon": attack strength
"pgd_iter": number of pgd iterations,
"pgd_step": PGD step size,
"pgd_test_restart_num": Number of random restarts.
"beta": Deprecated.
"test_start_idx": first dataset idx
"test_end_idx": last dataset idx
"idx_step_size": idx step size
```
The dataset samples that are taken are: range(test_start_idx, test_end_idx, idx_step_size)

## model parameters
```
"model_arch": model archietecture. For MNIST set MNISTClassifier, for CIFAR10 set RST for Carmon et al. [2019] model and wide_resnet for WRN model. For ImageNet set resnet50.
"ckpt_path": path to trained model
"pnml_active": Whether to use adversarial pNML scheme
```

## fit_to_sample parameters
```
"attack_type": "fgsm",
"epsilon": attack strength
"pgd_iter": number of pgd iterations,
"pgd_step": PGD step size,
"pgd_test_restart_num": Number of random restarts.
```

## initial_training parameters
These parameters are only used during training (when src\train.py is executed):
```
"eval_test_every_n_epoch": Perform testset evaluation when epochs % eval_test_every_n_epoch == 0
"save_model_every_n_epoch": Save model when epochs % eval_test_every_n_epoch == 0 
"epochs": Number of training epochs
"lr": Learning rate
"momentum": SGD Momentum
"step_size": A list that contains the number of epoch afterwhich lr is updated to be lr = lr * gamma
"gamma": The factor which multiply the learning rate
"weight_decay": L2 regularization ,
"loss_goal": Train until goal is reached. Set to 0.0 to train for all epochs,
"adv_alpha": The ratio between natural and adversarial loss. Set to 1.0 for robust training.
"adv_attack_train": The adversaril attack for the training set 
    {  
    "attack_type": "pgd",
    "epsilon": The strength of the attack,
    "pgd_iter": number of pgd iterations,
    "pgd_step": PGD step size,
    "pgd_test_restart_num": Number of random restarts.
    "beta": Deprecated.
    }
```

