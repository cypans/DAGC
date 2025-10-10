#! /bin/bash
dataset_name=PubMed
data_dir=./data
split_type=fixed_splits
device=cuda:0
epoch=1000
num_parts=5
batch_size=1
lr=0.01
wd=0.003
kernel_size1=3
kernel_size2=3
times=10
leaf=3
model_path=/ExtHDD/Users/pcy/DMAN/results/PubMed/79.7000.pth

python dman_visization.py --model_path $model_path --leaf $leaf --dataset_name $dataset_name --data_dir $data_dir --split_type $split_type --device $device --epoch $epoch --num_parts $num_parts --batch_size $batch_size --lr $lr --wd $wd --kernel_size1 $kernel_size1 --kernel_size2 $kernel_size2 --times $times