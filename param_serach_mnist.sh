#!/bin/bash
#dos2unix test1.sh
random_restart_num=(0)
fix_iter=(1)
fix_eps_arr=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08)
first_cuda_device_index=0
idx=0
array_len="${#fix_eps_arr[@]}"
for ((n=0; n<${#random_restart_num[@]}; ++n));
do
    for ((i=0; i<${#fix_iter[@]}; ++i));
    do
        for ((idx=0; idx<$((array_len-1)); ++idx));
        do
            cuda_device=$((first_cuda_device_index + idx))
            printf "\ncuda_device=%s" "$cuda_device"
            #if [ $idx -lt $((array_len-1)) ]
            CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t mnist_adversarial -r ${fix_eps_arr[idx]} -o output/mnist_param_search_pgd_1iter -i ${fix_iter[i]} -n ${random_restart_num[n]}&
            sleep 2
        done
        cuda_device=$((first_cuda_device_index + idx))
        CUDA_VISIBLE_DEVICES=$cuda_device python ./src/main.py -t mnist_adversarial -r ${fix_eps_arr[idx]} -o output/mnist_param_search_pgd_1iter -i ${fix_iter[i]} -n ${random_restart_num[n]}
    done
done