
import sys
import os
import argparse
import torch

sys.path.append(os.path.join(sys.path[0],'..','..','..','..'))

from rl_agent.hybrid_cp_rl_solver.problem.tsp.learning.trainer_dqn import TrainerDQN



os.environ['KMP_DUPLICATE_LIB_OK']='True'

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--n_city', type=int, default=20)
    parser.add_argument('--grid_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)

    # Hyper parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_step', type=int, default=-1)
    parser.add_argument('--max_softmax_beta', type=float, default=10, help="max_softmax_beta")
    parser.add_argument('--hidden_layer', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=128, help='dimension of latent layers')


    # Argument for Trainer
    parser.add_argument('--n_episode', type=int, default=1000000)
    parser.add_argument('--time_limit', type=int, default=48*60*60) # seconds
    parser.add_argument('--save_dir', type=str, default='./result-default')
    parser.add_argument('--plot_training', type=int, default=1)
    parser.add_argument('--mode', default='cpu', help='cpu/gpu')

    parser.add_argument('--onnx', type=int, default=1, help='1 if onnx compatible, 0 if pytorch native')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    args.mode = "gpu" if torch.cuda.is_available() else "cpu"

    print("***********************************************************")
    print("[INFO] TRAINING ON RANDOM INSTANCES: TSP")
    print("[INFO] n_city: %d" % args.n_city)
    print("[INFO] grid_size: %d" % args.grid_size)
    print("[INFO] seed: %s" % args.seed)
    print("***********************************************************")
    print("[INFO] TRAINING PARAMETERS")
    print("[INFO] algorithm: DQN")
    print("[INFO] batch_size: %d" % args.batch_size)
    print("[INFO] learning_rate: %f" % args.learning_rate)
    print("[INFO] hidden_layer: %d" % args.hidden_layer)
    print("[INFO] latent_dim: %d" % args.latent_dim)
    print("[INFO] softmax_beta: %d" % args.max_softmax_beta)
    print("[INFO] n_step: %d" % args.n_step)
    print("[INFO] n_episode: %d" % args.n_episode)
    print("[INFO] time_limit: %d" % args.time_limit)
    print("[INFO] device: %s" % args.mode)
    print("[INFO] onnx compatible model: %d" % args.onnx)
    print("***********************************************************")
    sys.stdout.flush()

    # If the save_dir does not exsit, make it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    trainer = TrainerDQN(args)
    trainer.run_training()
