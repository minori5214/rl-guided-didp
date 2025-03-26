
import sys
import os
import torch
import argparse

sys.path.append(os.path.join(sys.path[0],'..','..','..','..'))

from rl_agent.hybrid_cp_rl_solver.problem.portfolio.learning.trainer_dqn import TrainerDQN

os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_num_threads(1)
print("cpu thread used: ", torch.get_num_threads())
os.environ['MKL_NUM_THREADS'] = '1'

torch.cuda.set_per_process_memory_fraction(0.25)

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--n_item', type=int, default=10)
    parser.add_argument('--capacity_ratio', type=float, default=0.5)
    parser.add_argument('--lambda_1', type=int, default=1)
    parser.add_argument('--lambda_2', type=int, default=5)
    parser.add_argument('--lambda_3', type=int, default=5)
    parser.add_argument('--lambda_4', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)


    # Hyper parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0) # 0.0001
    parser.add_argument('--n_step', type=int, default=-1)
    parser.add_argument('--max_softmax_beta', type=int, default=10, help="max_softmax_beta")
    parser.add_argument('--hidden_layer', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=128, help='dimension of latent layers')


    # Argument for Trainer
    parser.add_argument('--n_episode', type=int, default=5000000)
    parser.add_argument('--time_limit', type=int, default=48*60*60) # seconds
    parser.add_argument('--save_dir', type=str, default='./result-default')
    parser.add_argument('--plot_training', type=int, default=1)
    parser.add_argument('--mode', default='cpu', help='cpu/gpu')

    parser.add_argument('--model', default='narita', help='cappart/narita') # only 'narita' is supported


    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    args.mode = "gpu" if torch.cuda.is_available() else "cpu"

    print("***********************************************************")
    print("[INFO] TRAINING ON RANDOM INSTANCES: portfolio")
    print("[INFO] n_items: %d" % args.n_item)
    print("[INFO] lambda_1: %d" % args.lambda_1)
    print("[INFO] lambda_2: %d" % args.lambda_2)
    print("[INFO] lambda_3: %d" % args.lambda_3)
    print("[INFO] lambda_4: %d" % args.lambda_4)
    print("[INFO] capacity_ratio: %f" % args.capacity_ratio)
    print("[INFO] seed: %s" % args.seed)
    print("***********************************************************")
    print("[INFO] TRAINING PARAMETERS")
    print("[INFO] algorithm: DQN")
    print("[INFO] batch_size: %d" % args.batch_size)
    print("[INFO] learning_rate: %f" % args.learning_rate)
    print("[INFO] hidden_layer: %d" % args.hidden_layer)
    print("[INFO] latent_dim: %d" % args.latent_dim)
    print("[INFO] softmax_beta: %d" % args.max_softmax_beta)
    print("[INFO] time_limit: %d" % args.time_limit)
    print("[INFO] n_step: %d" % args.n_step)
    print("***********************************************************")
    sys.stdout.flush()

    trainer = TrainerDQN(args)
    trainer.run_training()
