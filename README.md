# Reinforcement Learning-based Heuristics to Guide Domain-Independent Dynamic Programming

Domain-Independent Dynamic Programming (DIDP) is a state-space search paradigm based on dynamic programming for combinatorial optimization. In its current implementation, DIDP guides the search using user-defined dual bounds. Reinforcement learning (RL) is increasingly being applied to combinatorial optimization problems and shares several key structures with DP, being represented by the Bellman equation and state-based transition systems. We propose using reinforcement learning to obtain a heuristic function to guide the search in DIDP. We develop two RL-based guidance approaches: value-based guidance using Deep Q-Networks and policy-based guidance using Proximal Policy Optimization. Our experiments indicate that RL-based guidance significantly outperforms standard DIDP and problem-specific greedy heuristics with the same number of node expansions. Further, despite longer node evaluation times, RL guidance achieves better run-time performance than standard DIDP on three of four benchmark domains.

This repository contains the code and implementation for our recent paper on using Reinforcement Learning (RL) to guide Domain-Independent Dynamic Programming (DIDP), accepted at CPAIOR 2025. 

Please note that `rl_agent/hybrid_cp_rl_solver` is a clone of hybrid-cp-rl-solver (https://github.com/qcappart/hybrid-cp-rl-solver/tree/master) by Cappart et al., with minor modifications to the RL models. Also, `didp-rs-dev` is a development version of `didp-rs` (https://github.com/domain-independent-dp/didp-rs), developed by Kuroiwa et al. Full credits go to the original authors and developers of these repositories.

## Installation

### 1. Build the Rust Component
Ensure you have Python 3.12 and Rust installed. Then, build the Rust project in release mode:

```sh
cd didp-rs-dev
cargo build --release
```

### 2. Build the Python Package
Move into the `didppy` directory and set up a virtual environment using Python 3.12:

```sh
cd didppy
python3.12 -m venv .venv
source .venv/bin/activate
```

Install `maturin`:

```sh
python3.12 -m pip install maturin
```

Build the `didppy` package:

```sh
maturin build --release --manylinux off
```

Deactivate the virtual environment:

```sh
deactivate
```

### 3. Install the `didppy` Package
Create another virtual environment to install `didppy`:

```sh
python3.12 -m venv didppy-release
```

Activate the new virtual environment:

```sh
source didppy-release/bin/activate
```

Install `didppy` from the locally built wheel file:

```sh
pip install ../target/wheels/didppy-0.7.2-cp37-abi3-linux_x86_64.whl
```

After installation, deactivate the environment:

```sh
deactivate
```

Move back to the root directory:

```sh
cd ../../
```

### 4. Install Dependencies
### Install PyTorch and DGL (Deep Graph Library) (GPU version)
If using a GPU, install PyTorch with CUDA 11.8 support:

```sh
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

Install DGL with CUDA 11.8 support:

```sh
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html
```

To install additional dependencies:

```sh
pip install -r requirements.txt
```

After installation, deactivate the environment:

```sh
deactivate
```


### Install PyTorch and DGL (Deep Graph Library) (CPU only)
If GPU support is not available, install the CPU-only versions:

```sh
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/repo.html
pip install -r requirements_cpu.txt
```

After installation, deactivate the environment:

```sh
deactivate
```


## Usage

### 1. Activate the Virtual Environment

```sh
source ./didp-rs-dev/didppy/didppy-release/bin/activate
```

### 2. Training a model

If no GPU is detected, the training will automatically run on the CPU.

```sh
./run_training_dqn_tsp.sh # DQN
./run_training_dqn_tsp.sh # PPO
```

### 3. Solve instances

#### TSP
```
./solve_tsp.sh --n 20 --heuristic dqn --policy-name none --solver-name CABS
```


#### TSPTW
```
./solve_tsptw.sh --n 20 --heuristic dqn --policy-name none --solver-name CABS
```

#### Portfolio
```
./solve_portfolio.sh --n 20 --heuristic dqn --policy-name none --solver-name CABS
```

## Citation
If you use this repository in your research, please cite our paper:

- **arXiv version:** [https://arxiv.org/abs/2503.16371](https://arxiv.org/abs/2503.16371)
- **CPAIOR 2025:** TBA

```
@article{narita2025reinforcement,
  title={Reinforcement Learning-based Heuristics to Guide Domain-Independent Dynamic Programming},
  author={Narita, Minori and Kuroiwa, Ryo and Beck, J Christopher},
  journal={arXiv preprint arXiv:2503.16371},
  year={2025}
}
```

---

