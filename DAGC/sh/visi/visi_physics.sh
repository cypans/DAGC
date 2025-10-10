#! /bin/bash
dataset_name=coauthor-physics
data_dir=./data
split_type=fixed_splits
device=cuda:0
epoch=1000
num_parts=5
batch_size=1
lr=0.02
wd=1e-8  #8
kernel_size1=2 #3
kernel_size2=1 #3
times=10
leaf=1.0
model_path='./results/coauthor-physics/97.40579710144928.pth'

python dman_visization.py --model_path $model_path --leaf $leaf --dataset_name $dataset_name --data_dir $data_dir --split_type $split_type --device $device --epoch $epoch --num_parts $num_parts --batch_size $batch_size --lr $lr --wd $wd --kernel_size1 $kernel_size1 --kernel_size2 $kernel_size2 --times $times