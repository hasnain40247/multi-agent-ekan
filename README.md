# Equivariant Kolmogorov-Arnold Networks for Multi-Agent Coordination

# Overview
This project addresses a fundamental challenge in multi-agent coordination: traditional neural network policies fail to exploit inherent symmetries present in cooperative multi-agent systems. Agents are often interchangeable (permutation symmetry), and optimal strategies depend on relative positions rather than absolute coordinates (geometric symmetry).

## Environment
We use the Multi-Agent Particle Environment (MPE) from PettingZoo, specifically the Cooperative Navigation task where:

- N agents must simultaneously cover N landmarks
- Agents must avoid collisions with each other
- Rewards encourage coverage while penalizing collisions

EKANS code from [here](https://github.com/hulx2002/EKAN).


## Installation

### Setup Steps

1. **Clone the repo**

2. **Create a virtual environment, tested using python 3.10.5:**
   ```bash
   python -m venv .venv
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Install Adan for ekan to work**
    ```bash
    git clone https://github.com/sail-sg/Adan.git
    cd Adan
    python3 setup.py install --unfused
    ```

## Usage

The main script is executed as a Python module with config file to run experiments. Use `config.py` to update configs on training.

### Basic Command (To Run EKAN)

```bash
python train.py --actor "[traditional,ekan]" --critic "[traditional,permutation invariant]"
```

Arguments that can be provided to the actor include:
 - traditional
 - ekan
 - "rotational equivariant"

Arguments that can be provided to the critic include.
- traditional
- "permutation invariant"

### Basic Command (To Run EGNN/EMLP)

```bash
cd egnn_emlp_actor_variants
```
Here, we have our ```main.py``` which has custom hyperparameters as well as argumnents that you can change to run specific variants.

```bash
python main.py --actor_type "[mlp,emlp,egnn]" --critic_type "[mlp,gcn_max]"
```