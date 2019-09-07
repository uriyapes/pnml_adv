#!/bin/bash
#dos2unix test1.sh
fix_eps_arr=(0.001 0.005 0.01 0.03 0.05 0.08 0.011)
first_cuda_device_index=0
idx=0
array_len="${#fix_eps_arr[@]}"
for ((idx=0; idx<$((array_len - 1)); ++idx));
do
    cuda_device=$((first_cuda_device_index + idx))
    printf "\ncuda_device=%s" "$cuda_device"
    CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t cifar_adversarial -e ${fix_eps_arr[idx]} -o output/cifar_diff_fix_natural &
    sleep 2
done
cuda_device=$((first_cuda_device_index + idx))
CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t cifar_adversarial -e ${fix_eps_arr[idx]} -o output/cifar_diff_fix_natural