#! /bin/bash
dataset_name=CiteSeer
data_dir=./data
split_type=fixed_splits
device=cuda:0
epoch=1500
num_parts=5
batch_size=1
lr=0.001
wd=0.02
kernel_size1=6
kernel_size2=4
times=10
leaf=3
model_path=/ExtHDD/Users/pcy/DMAN/results/CiteSeer/74.0000.pth

python dman_visization.py --model_path $model_path --leaf $leaf --dataset_name $dataset_name --data_dir $data_dir --split_type $split_type --device $device --epoch $epoch --num_parts $num_parts --batch_size $batch_size --lr $lr --wd $wd --kernel_size1 $kernel_size1 --kernel_size2 $kernel_size2 --times $times