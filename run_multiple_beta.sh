#!/bin/bash
#dos2unix test1.sh
exp_name="mnist_adversarial"
beta_arr=(0 0.0005 0.001 0.005 0.01 0.05 0.1 0.25 0.36 0.5 0.66 0.8 1)
#beta_arr=(0.01 0.02 0.025 0.0313 0.036 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11)
first_cuda_device_index=0
idx=0
array_len="${#beta_arr[@]}"
for ((idx=0; idx<$((array_len - 1)); ++idx));
do
    cuda_device=$((first_cuda_device_index + idx))
    printf "\ncuda_device=%s" "$cuda_device"
    CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t $exp_name -b ${beta_arr[idx]} -o output/mnist_beta_search_lambda_01_bpda_model &
    sleep 4
done
cuda_device=$((first_cuda_device_index + idx))
CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t $exp_name -b ${beta_arr[idx]} -o output/mnist_beta_search_lambda_01_bpda_model