#!/bin/bash

# Seed for the random generation: ensure that the validation set remains the same.
seed=2

# Characterics of the training instances
n_item=50
capacity_ratio=0.5
lambda_1=1
lambda_2=5
lambda_3=5
lambda_4=5

# cappart, narita
model='narita'

# Parameters for the training
batch_size=128 # max batch size for training/testing
hidden_layer=3 # number of hidden layer
latent_dim=256
learning_rate=0.00001
n_step=-1
max_softmax_beta=10

# Others
plot_training=1 # Boolean value: plot the training curve or not
mode=gpu

# Folder to save the trained model
network_arch=hidden_layer-$hidden_layer-latent_dim-$latent_dim/
result_root=rl_agent/hybrid_cp_rl_solver/trained-models/dqn/portfolio/n-item-$n_item/capacity_ratio-$capacity_ratio/lambdas-$lambda_1-$lambda_2-$lambda_3-$lambda_4/seed-$seed/$network_arch
save_dir=$result_root/batch_size-$batch_size-learning_rate-$learning_rate-n_step-$n_step-max_softmax_beta-$max_softmax_beta


if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python rl_agent/hybrid_cp_rl_solver/problem/portfolio/main_training_dqn_portfolio.py \
    --seed $seed  \
    --n_item $n_item  \
    --capacity_ratio $capacity_ratio \
    --lambda_1 $lambda_1 \
    --lambda_2 $lambda_2 \
    --lambda_3 $lambda_3 \
    --lambda_4 $lambda_4 \
    --batch_size $batch_size  \
    --hidden_layer $hidden_layer  \
    --latent_dim $latent_dim  \
    --max_softmax_beta $max_softmax_beta \
    --learning_rate $learning_rate \
    --save_dir $save_dir  \
    --plot_training $plot_training  \
    --mode $mode \
    --n_step $n_step \
    --model $model \
    2>&1 | tee $save_dir/log-training.txt


