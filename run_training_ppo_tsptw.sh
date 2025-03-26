#!/bin/bash

# Seed for the random generation: ensure that the validation set remains the same.
seed=1

# Characterics of the training instances
n_city=20
grid_size=100
max_tw_gap=100
max_tw_size=1000

# Parameters for the training
k_epochs=3
update_timestep=2048
learning_rate=0.0001
entropy_value=0.001
eps_clip=0.1
batch_size=128 # batch size must be a divisor of update_timestep
latent_dim=256
hidden_layer=4

# Others
plot_training=1 # Boolean value: plot the training curve or not
mode=gpu

# Folder to save the trained model
network_arch=hidden_layers-$hidden_layer-latent_dim-$latent_dim/
result_root=rl_agent/hybrid_cp_rl_solver/trained-models/ppo/tsptw/n$n_city/grid-$grid_size-tw-$max_tw_gap-$max_tw_size/seed-$seed/$network_arch
save_dir=$result_root/kep$k_epochs-ut$update_timestep-bs$batch_size-lr$learning_rate-evalue$entropy_value-eclip$eps_clip


if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python rl_agent/hybrid_cp_rl_solver/problem/tsptw/main_training_ppo_tsptw.py \
    --seed $seed  \
    --n_city $n_city  \
    --grid_size $grid_size \
    --max_tw_gap $max_tw_gap \
    --max_tw_size $max_tw_size \
    --k_epochs $k_epochs \
    --update_timestep $update_timestep \
    --learning_rate $learning_rate \
    --eps_clip $eps_clip \
    --entropy_value $entropy_value \
    --batch_size $batch_size \
    --latent_dim $latent_dim  \
    --hidden_layer $hidden_layer  \
    --save_dir $save_dir  \
    --plot_training $plot_training  \
    --mode $mode \
    2>&1 | tee $save_dir/log-training.txt


