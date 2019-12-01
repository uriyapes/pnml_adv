#!/bin/bash
#dos2unix test1.sh
exp_name="mnist_adversarial"
#exp_name="cifar_adversarial"
test_eps_arr=(0.05 0.1 0.15 0.2 0.25 0.3 0.33 0.36 0.39 0.42 0.45 0.5)
#test_eps_arr=(0.01 0.02 0.025 0.0313 0.036 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11)
first_cuda_device_index=0
idx=0
array_len="${#test_eps_arr[@]}"
for ((idx=0; idx<$((array_len - 1)); ++idx));
do
    cuda_device=$((first_cuda_device_index + idx))
    printf "\ncuda_device=%s" "$cuda_device"
    CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t $exp_name -e ${test_eps_arr[idx]} -o output/mnist_bpda_model_diff_eps_pgd_50iter &
    sleep 4
done
cuda_device=$((first_cuda_device_index + idx))
CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t $exp_name -e ${test_eps_arr[idx]} -o output/mnist_bpda_model_diff_eps_pgd_50iter