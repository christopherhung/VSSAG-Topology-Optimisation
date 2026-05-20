# VSSAG-Topology-Optimisation
# Overcoming the Discrete-Continuous Trilemma in Topology Optimisation

This repository contains the official PyTorch implementation of the **Variational State-Space Augmented Graph (VSSAG)** framework, as presented in the thesis: *[Overcoming the Discrete-Continuous Trilemma in Topology Optimisation using Exact Discrete Adjoints and Augmented Lagrangians]*.

## Core Architectural Features
* **Exact Discrete Adjoint CPU-Bypass:** Achieves a 0.0 MB GPU VRAM footprint for physics evaluation, completely bypassing the $\mathcal{O}(L \cdot D)$ memory wall inherent to standard Physics-Informed Neural Networks.
* **Hestenes-Powell ALM Loop:** Enforces strict physical manufacturability, dynamically eradicating "acoustic foam" intermediate densities via a conditional Heaviside projection.
* **Stabilised Factorised HyperSIREN:** Maps discrete bipartite graphs to a continuous spatial manifold without suffering from Hypernetwork parameter explosion or high-frequency spectral collapse.

## Execution Instructions
To reproduce the empirical telemetry and topological results:

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. To train the models and generate the final topologies: `python train.py`
4. To reproduce the Memory Wall hardware stress test: `python script_scaling_test.py`
