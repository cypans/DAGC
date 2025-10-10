#! /bin/bash
dataset_name=coauthor-cs
data_dir=./data
split_type=fixed_splits
device=cuda:0
epoch=1500
num_parts=5
batch_size=1
lr=0.15
wd=1e-8
kernel_size1=2 #3
kernel_size2=1 #3
times=5
leaf=1.5
hidden_feat_dim=128

python main.py --hidden_feat_dim $hidden_feat_dim --leaf $leaf --dataset_name $dataset_name --data_dir $data_dir --split_type $split_type --device $device --epoch $epoch --num_parts $num_parts --batch_size $batch_size --lr $lr --wd $wd --kernel_size1 $kernel_size1 --kernel_size2 $kernel_size2 --times $times