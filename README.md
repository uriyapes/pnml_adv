# deep_pnml


This is the official implementioation of Universal Learning Approach for Adversarial Defense paper.


Get started:

1. Clone the repository

2. Intsall requeirement 

'''
pip install -r requirements.txt
'''

3. Run basic experimnet:

'''
CUDA_VISIBLE_DEVICES=0 python src/main.py -t mnist_adversarial
'''

## Experimnets:

The experimnet options are:

1. mnist_adversarial: running adversarial pNML on CIFAR10 dataset.
2. mnist_adversarial: running adversarial pNML on MNIST dataset.


The parameters of each experimnet can be change in the parameters file: src\params.json
The src/params.json file contains separate parameters for each experiment, i.e. for 
mnist_adversarial the parameters are:

    "mnist_adversarial": {
        "batch_size": 128,
        "num_workers": 4,
        "freeze_layer": 0,
        "adv_attack_test": {
            ...
        },
        "initial_training": {
            ...
        },
        "fit_to_sample": {
            ...
        }
    }
The adv_attack_test section describes the attack, the initial_training describes 
the training parameters of the model and fit_to_sample describes the refinement parameters.




