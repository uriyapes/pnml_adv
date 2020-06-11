# Adversarial pNML

This is the official implementation of "A UNIVERSAL LEARNING APPROACH FOR ADVERSARIAL DEFENSE" paper.


## Requirements:

1. Clone the repository.

2. Create conda enviorment: conda env create -f environment.yml

3. Activate conda environment.yml: conda activate pnml_adv3

4. Download CIFAR10 model from: 

    https://github.com/yaircarmon/semisup-adv.
    
    Download ImageNet model from:
     
    https://github.com/locuslab/fast_adversarial. 

5. Edit the json parameter files within /src/parameters:
Inside "model" change "ckpt_path" to the path of the downloaded models.

6. Download ImageNet validation set http://www.image-net.org/challenges/LSVRC/2012/downloads into ./data/imagenet/val directory:
 
     
## Evaluate the model:
### White-box attack
#### Base model
The following command will evaluate the base model robustness against PGD attack:
```
python src/eval.py -o <output_dir_path> -t <experiment_type> -p <params_path> --general.save <bool>
# For MNIST:
python src/eval.py -o ./output -t mnist_adversarial -p ./src/parameters/mnist_params.json
# For CIFAR10:
python src/eval.py -o ./output -t cifar_adversarial -p ./src/parameters/cifar_params.json
# For ImageNet:
python src/eval.py -o ./output -t imagenet_adversarial -p ./src/parameters/imagenet_params.json
```
 
* output_dir_path - the location of the output of the script.
* experiment_type - Type of experiment, must be one of the following: 'mnist_adversarial', 'cifar_adversarial', 'imagenet_adversarial'.
* params_path - A path to a json parameter file that contains the experiment parameters.
* general.save - Whether to save the generated adversarial samples. Default value is False.

The specific details of the experiment are given in the json parameter file.

To evaluate natural accuracy use adv_attack_test.attack_type argument, for example, for MNIST:
```
python src/eval.py -o ./output -t mnist_adversarial -p ./src/parameters/mnist_params.json --adv_attack_test.attack_type natural
```
#### Adversarial pNML
To evaluate adversarial pNML (with any base model) change the field "model"->"pnml_active" to true
in the json parameter file. 
For example, to evaluate adversarial pNML robustness against adaptive attack run:
```
python src/eval.py -o ./output -t mnist_adversarial -p ./src/parameters/mnist_params_pnml_adaptive.json
```

To evaluate adversarial pNML against PGD attack we first need to generate PGD adversarial samples
using the base model and save them into adversarials.t file. Then, evaluate the samples for adversarial pNML scheme. 
To generate adversarials.t file that hold adversarial samples set general.save to true and run eval.py:
```
python src/eval.py -o ./output -t mnist_adversarial -p ./src/parameters/mnist_params.json --general.save true
```
Then, locate adversarials.t in the output folder and update the json parameter file:
* Set adv_attack_test->black_box_adv_path to  adversarials.t path.
* Set adv_attack_test->attack_type to "natural".
* Make sure model->pnml_active is true.
Run eval.py with the updated param file:
```
python src/eval.py -o ./output -t mnist_adversarial -p ./src/parameters/mnist_params_pnml_pgd.json
```
### Black-box attack
To evaluate HSJA attack run:
```
python src/eval_hsj.py -o <output_dir_path> -t <experiment_type> -p <params_path>
```


## Training:
```
python src/train.py -o <output_dir_path> -t <experiment_type> -p <params_path>
# For toy dataset:
python src/train.py -o ./output -t synthetic -p ./src/parameters/synthetic_params.json
# For MNIST:
python src/train.py -o ./output -t mnist_adversarial -p ./src/parameters/mnist_params.json
# For CIFAR10:
python src/train.py -o ./output -t cifar_adversarial -p ./src/parameters/cifar_params.json
```
  

