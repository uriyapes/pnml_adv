#!/bin/bash
#dos2unix test1.sh
#Search for Beta value
beta_arr=(0.001 0.005 0.01 0.025 0.5 1)
fix_iter=(1)
fix_eps_arr=(0.04)
first_cuda_device_index=0
idx=0
array_len="${#beta_arr[@]}"
for ((n=0; n<${#fix_eps_arr[@]}; ++n));
do
    for ((i=0; i<${#fix_iter[@]}; ++i));
    do
        for ((idx=0; idx<$((array_len-1)); ++idx));
        do
            cuda_device=$((first_cuda_device_index + idx))
            printf "\ncuda_device=%s" "$cuda_device"
            #if [ $idx -lt $((array_len-1)) ]
            CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t mnist_adversarial -r ${fix_eps_arr[n]} -o output/mnist_lambda_005_beta_search -i ${fix_iter[i]} -b ${beta_arr[idx]}&
            sleep 2
        done
        cuda_device=$((first_cuda_device_index + idx))
        CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t mnist_adversarial -r ${fix_eps_arr[n]} -o output/mnist_lambda_005_beta_search -i ${fix_iter[i]} -b ${beta_arr[idx]}
    done
done