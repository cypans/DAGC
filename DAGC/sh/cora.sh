#! /bin/bash
dataset_name=Cora
data_dir=./data
split_type=fixed_splits
device=cuda:0
epoch=1000
num_parts=5
batch_size=1
lr=0.001
wd=1e-2
kernel_size1=5
kernel_size2=4
hidden_feat_dim=64
times=1
leaf=3

python main.py --hidden_feat_dim $hidden_feat_dim --leaf $leaf --dataset_name $dataset_name --data_dir $data_dir --split_type $split_type --device $device --epoch $epoch --num_parts $num_parts --batch_size $batch_size --lr $lr --wd $wd --kernel_size1 $kernel_size1 --kernel_size2 $kernel_size2 --times $times