# deep_pnml


This is the official implementioation of TBD


Get started:

1. Clone the repository

2. Intsall requeirement 

'''
pip install -r requirements.txt
'''

3. Run basic experimnet:

'''
CUDA_VISIBLE_DEVICES=0 python src/main.py -t pnml_cifar10
'''

## Experimnets:

The experimnet options are:

1. pnml_cifar10: running pNML on CIFAR10 dataset.
2. random_labels: runing pNML on CIFAR10 dataset that its labels are random.
3. out_of_dist_svhn: trainset is CIFAR10. Execute pNML on SVHN dataset.
4. out_of_dist_noise:  trainset is CIFAR10. Execute pNML on Noise images.
5. pnml_mnist: runining pNML on MNIST dataset.

The parameters of each experimnet can be change in the parameters file: src\params.json
