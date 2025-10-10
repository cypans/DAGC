#! /bin/bash
dataset_name=wikics
data_dir=./data
split_type=fixed_splits
device=cuda:0
epoch=1500
num_parts=5
batch_size=1
lr=0.01
wd=1e-8
kernel_size1=2
kernel_size2=2
times=1
leaf=1.0

python main.py --leaf $leaf --dataset_name $dataset_name --data_dir $data_dir --split_type $split_type --device $device --epoch $epoch --num_parts $num_parts --batch_size $batch_size --lr $lr --wd $wd --kernel_size1 $kernel_size1 --kernel_size2 $kernel_size2 --times $times