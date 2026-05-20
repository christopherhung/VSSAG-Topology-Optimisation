"""
Centralised Hyperparameter Configuration for the VSSAG Framework.
Ensures absolute reproducibility
"""
from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    nelx: int = 100
    nely: int = 100
    vol_target: float = 0.40
    E0: float = 1.0           # Young's Modulus (Solid)
    Emin: float = 1e-9        # Young's Modulus (Void - Singularity Guard)
    penal: float = 3.0        # SIMP Penalization Power
    nu: float = 0.3           # Poisson's Ratio

@dataclass
class NetworkConfig:
    latent_dim_D: int = 64    # Global Latent Vector Dimension
    hypernet_rank_R: int = 8  # Low-Rank Matrix Factorization bottleneck
    siren_hidden_H: int = 128 # Continuous Manifold Width
    siren_omega_0: float = 30.0 # Spatial Frequency Base
    vssag_layers_L: int = 5   # Number of RK4 Pseudo-Time Steps
    vssag_dt: float = 0.05    # ODE Integration Step Size

@dataclass
class OptimizationConfig:
    epochs: int = 1500
    learning_rate: float = 2e-3
    adam_betas: tuple = (0.9, 0.999)
    
    # Augmented Lagrangian (ALM) & Heaviside Parameters
    alm_initial_mu: float = 50.0
    alm_max_mu: float = 500.0
    heaviside_initial_beta: float = 1.0
    heaviside_max_beta: float = 64.0
    continuation_multiplier: float = 1.2
    update_interval_epochs: int = 50
    volume_tolerance: float = 0.02

# Global Instantiation
PHYSICS = PhysicsConfig()
NETWORK = NetworkConfig()
OPTIM = OptimizationConfig()