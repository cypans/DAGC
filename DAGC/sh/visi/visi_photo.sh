#! /bin/bash
dataset_name=amazon-photo
data_dir=./data
split_type=fixed_splits
device=cuda:0
epoch=3000
num_parts=5
batch_size=1
lr=0.1
wd=1e-5
kernel_size1=1 #3
kernel_size2=2 #3
times=1
leaf=1.0
hidden_feat_dim=64
model_path=/ExtHDD/Users/pcy/DMAN/results/amazon-photo/96.2092.pth

python dman_visization.py --model_path $model_path --hidden_feat_dim $hidden_feat_dim --leaf $leaf --dataset_name $dataset_name --data_dir $data_dir --split_type $split_type --device $device --epoch $epoch --num_parts $num_parts --batch_size $batch_size --lr $lr --wd $wd --kernel_size1 $kernel_size1 --kernel_size2 $kernel_size2 --times $times