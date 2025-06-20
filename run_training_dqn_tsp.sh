#!/bin/bash

# Seed for the random generation: ensure that the validation set remains the same.
seed=2

# Characterics of the training instances
n_city=50
grid_size=100

# Parameters for the training
batch_size=128 # max batch size for training/testing # 256
hidden_layer=3 # number of hidden layer # 3
latent_dim=64 # 64
learning_rate=0.0001
n_step=-1
max_softmax_beta=2

# Others
plot_training=1 # Boolean value: plot the training curve or not
mode=gpu # cpu or gpu
onnx=1

# Folder to save the trained model
network_arch=hidden_layer-$hidden_layer-latent_dim-$latent_dim/
result_root=rl_agent/hybrid_cp_rl_solver/trained-models/dqn/tsp/n-city-$n_city/grid-$grid_size/seed-$seed/$network_arch
save_dir=$result_root/batch_size-$batch_size-learning_rate-$learning_rate-n_step-$n_step-max_softmax_beta-$max_softmax_beta


if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python rl_agent/hybrid_cp_rl_solver/problem/tsp/main_training_dqn_tsp.py \
    --seed $seed  \
    --n_city $n_city  \
    --grid_size $grid_size \
    --batch_size $batch_size  \
    --hidden_layer $hidden_layer  \
    --latent_dim $latent_dim  \
    --max_softmax_beta $max_softmax_beta \
    --learning_rate $learning_rate \
    --save_dir $save_dir  \
    --plot_training $plot_training  \
    --mode $mode \
    --n_step $n_step \
    --onnx $onnx \
    2>&1 | tee $save_dir/log-training.txt


