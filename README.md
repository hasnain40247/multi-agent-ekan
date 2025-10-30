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
├── environment/             
│   ├── __pycache__/
│   ├── __init__.py
│   └── mpe.py               # Multi-agent simple_spread_v3
├── mappo/                  
│   ├── __pycache__/
│   ├── __init__.py
│   └── mappo.py             # MAPPO framework
├── policy_network/           
│   ├── __pycache__/
│   ├── __init__.py
│   └── policy_net.py        # Basic policy net architecture
├── main.py                   # Main entry point
├── README.md                 
└── requirements.txt          # Python dependencies
```

## Installation

### Setup Steps

1. **Clone the repo**

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main script is executed as a Python module with configurable cli arguments.

### Basic Command

```bash
python -m main [OPTIONS]
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n_agents` | int | 3 | Number of agents in the environment |
| `--max_cycles` | int | 25 | Maximum number of cycles per episode |
| `--continuous` | flag | False | Use continuous action space instead of discrete |
| `--steps` | int | 50 | Number of visualization/training steps |
| `--delay` | float | 0.0 | Delay between frames in seconds (for visualization) |
| `--manual` | flag | False | Enable manual control mode for interactive testing |

