# Equivariant Kolmogorov-Arnold Networks for Multi-Agent Coordination

# Overview
This project addresses a fundamental challenge in multi-agent coordination: traditional neural network policies fail to exploit inherent symmetries present in cooperative multi-agent systems. Agents are often interchangeable (permutation symmetry), and optimal strategies depend on relative positions rather than absolute coordinates (geometric symmetry).

## Environment
We use the Multi-Agent Particle Environment (MPE) from PettingZoo, specifically the Cooperative Navigation task where:

- N agents must simultaneously cover N landmarks
- Agents must avoid collisions with each other
- Rewards encourage coverage while penalizing collisions


## Project Structure

```
.
├── configs
│   └── experiment.yaml
├── environment/             
│   ├── __init__.py
│   └── mpe.py                # Multi-agent simple_spread_v3
├── mappo/                  
│   ├── __init__.py
│   └── mappo.py              # MAPPO framework
├── policy_networks/           
│   ├── __init__.py
│   ├── kan_policy_net.py     # vanilla kan policy network
│   ├── mlp_policy_net.py     # mlp policy network
│   └── registry.py        
├── main.py                   # Main entry point
├── README.md                 
└── requirements.txt          # Python dependencies
```

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

## Usage

The main script is executed as a Python module with config file to run experiments.

### Basic Command

```bash
python -m main --config configs/experiment.yaml --override run.mode=visualize
```

## YAML Configuration

All runs are configured via a single YAML file (e.g., `configs/experiment.yaml`). You can override any field at launch with `--override section.key=value`. Booleans/numbers/null use JSON literals: `true/false/null`.

### `run` (session controls)

| Key      | Type | Default | Description |
|---|---|---|---|
| `mode`   | enum | `train` | `train` to learn; `visualize` to render env. |
| `seed`   | int  | `42`    | Global RNG seed (numpy/torch, CUDA if avail). |

**Example override:** `--override run.mode=visualize`

---

### `env` (environment setup)

| Key                   | Type  | Default | Description |
|---|---|---|---|
| `n_agents`            | int   | `3`     | Number of agents in the MPE scenario. |
| `max_cycles`          | int   | `25`    | Max steps per episode before truncation. |
| `continuous_actions`  | bool  | `true`  | Use continuous action space (MPE Box(0,1,…)). |
| `visualize.steps`     | int   | `50`    | Steps to run when `run.mode=visualize`. |
| `visualize.delay`     | float | `0.0`   | Seconds to sleep between frames in visualize. |
| `visualize.manual`    | bool  | `false` | If supported, allow manual interaction. |

**Example override:** `--override env.n_agents=5 --override env.max_cycles=50`

---

### `training` (learning hyperparameters)

| Key               | Type  | Default  | Description |
|---|---|---|---|
| `epochs`          | int   | `100`    | Number of training epochs. |
| `steps_per_epoch` | int   | `1024`   | Env steps collected per epoch (across all agents). |
| `gamma`           | float | `0.99`   | Discount factor. |
| `lr`              | float | `0.0003` | Adam learning rate. |
| `value_coef`      | float | `0.5`    | Weight on value-loss term. |
| `entropy_coef`    | float | `0.01`   | Entropy bonus weight (pre-tanh Normal entropy). |
| `max_grad_norm`   | float | `0.5`    | Global gradient-norm clip. |

**Example override:** `--override training.epochs=300 --override training.lr=1e-4`

---

### `model` (policy backbone selection & hyperparameters)

Top-level fields under `model` act as **shared defaults** across all backbones. Each backbone may also define its **own** sub-block that can override shared defaults.

| Key           | Type | Default | Description |
|---|---|---|---|
| `name`        | enum | `mlp`   | Active backbone: `mlp` \| `kan` \| (extend via registry). |
| `hidden_dim`  | int  | `64`    | Shared default hidden width (overridden by sub-block fields). |

#### `model.mlp` (when `name: mlp`)
| Key | Type | Default | Description |
|---|---|---|---|
| `hidden_dim`  | int  | `128`    | Overrides shared width for the MLP. |



#### `model.kan` (when `name: kan`)
| Key          | Type | Default | Description |
|---|---|---|---|
| `hidden_dim` | int  | `64`    | Overrides shared width for the KAN backbone. |
| `grid`       | int  | `16`    | Spline grid size (pykan hyperparameter). |
| `k`          | int  | `3`     | Spline order/degree-like hyperparameter. |
| `kw`         | dict | `{}`    | Extra kwargs passed to `pykan.KAN` constructor. |

**Switch backbones:** `--override model.name=kan`  
**KAN tuning:** `--override model.kan.grid=32 --override model.kan.k=3`

---

### `wandb` (experiment tracking)

| Key       | Type        | Default             | Description |
|---|---|---|---|
| `enabled` | bool        | `true`              | Toggle Weights & Biases logging. |
| `project` | str         | `multi-agent-ekan`  | W&B project name. |
| `name`    | str \| null | `null`              | Run name (auto if null). |
| `entity`  | str \| null | `null`              | Team/entity (optional). |

**Example override:** `--override wandb.enabled=false`