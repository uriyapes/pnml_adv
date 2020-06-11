#!/bin/bash
#dos2unix test1.sh
fix_eps_arr=(0.00392156862 0.007842 0.01176 0.0156862745098 0.0196078431 0.03137254896 0.15)
first_cuda_device_index=0
idx=0
array_len="${#fix_eps_arr[@]}"
for ((idx=0; idx<$((array_len - 1)); ++idx));
do
    cuda_device=$((first_cuda_device_index + idx))
    printf "\ncuda_device=%s" "$cuda_device"
    CUDA_VISIBLE_DEVICES=$cuda_device python ./src/eval.py -t imagenet_adversarial -r ${fix_eps_arr[idx]} -o output/imagenet_diff_fix_pgd &
    sleep 4
done
cuda_device=$((first_cuda_device_index + idx))
CUDA_VISIBLE_DEVICES=$cuda_device python ./src/eval.py -t imagenet_adversarial -r ${fix_eps_arr[idx]} -o output/imagenet_diff_fix_pgd