# Equivariant Kolmogorov-Arnold Networks for Multi-Agent Coordination

A project implementing Equivariant Kolmogorov-Arnold Networks for cooperative multi-agent reinforcement learning, with a focus on exploiting inherent symmetries in multi-agent systems.

# Overview
This project addresses a fundamental challenge in multi-agent coordination: traditional neural network policies fail to exploit inherent symmetries present in cooperative multi-agent systems. Agents are often interchangeable (permutation symmetry), and optimal strategies depend on relative positions rather than absolute coordinates (geometric symmetry).

## Environment
We use the Multi-Agent Particle Environment (MPE) from PettingZoo, specifically the Cooperative Navigation task where:

- N agents must simultaneously cover N landmarks
- Agents must avoid collisions with each other
- Rewards encourage coverage while penalizing collisions
