#! /bin/bash
dataset_name=amazon-computer
data_dir=./data
split_type=fixed_splits
device=cuda:0
epoch=3000
num_parts=5
batch_size=1
lr=0.1
wd=7e-6
kernel_size1=2 #3
kernel_size2=2 #3
times=1
leaf=1
hidden_feat_dim=64

python main.py --hidden_feat_dim $hidden_feat_dim --leaf $leaf --dataset_name $dataset_name --data_dir $data_dir --split_type $split_type --device $device --epoch $epoch --num_parts $num_parts --batch_size $batch_size --lr $lr --wd $wd --kernel_size1 $kernel_size1 --kernel_size2 $kernel_size2 --times $times