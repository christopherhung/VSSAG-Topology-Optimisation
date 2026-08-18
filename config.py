"""
Centralised Hyperparameter Configuration for the VSSAG Framework.
Powered by Pydantic for strict runtime type validation.
"""
from pydantic import BaseModel, Field
from typing import Tuple

class PhysicsConfig(BaseModel):
    nelx: int = Field(100, description="Mesh resolution X")
    nely: int = Field(100, description="Mesh resolution Y")
    vol_target: float = Field(0.40, description="Target volume fraction constraint")
    E0: float = Field(1.0, description="Young's Modulus (Solid)")
    Emin: float = Field(1e-9, description="Young's Modulus (Void - Singularity Guard)")
    penal: float = Field(3.0, description="SIMP Penalization Power")
    nu: float = Field(0.3, description="Poisson's Ratio")

class NetworkConfig(BaseModel):
    latent_dim_D: int = Field(64, description="Global Latent Vector Dimension")
    hypernet_rank_R: int = Field(8, description="Low-Rank Matrix Factorization bottleneck")
    siren_hidden_H: int = Field(128, description="Continuous Manifold Width")
    siren_omega_0: float = Field(30.0, description="Spatial Frequency Base")
    vssag_layers_L: int = Field(5, description="Number of RK4 Pseudo-Time Steps")
    vssag_dt: float = Field(0.05, description="ODE Integration Step Size")

class OptimizationConfig(BaseModel):
    epochs: int = Field(1500, description="Total training epochs")
    learning_rate: float = Field(2e-3, description="Adam optimizer learning rate")
    adam_betas: Tuple[float, float] = Field((0.9, 0.999), description="Adam betas")
    alm_initial_mu: float = Field(50.0, description="Initial ALM penalty")
    alm_max_mu: float = Field(500.0, description="Maximum ALM penalty")
    heaviside_initial_beta: float = Field(1.0, description="Initial Heaviside parameter")
    heaviside_max_beta: float = Field(64.0, description="Maximum binarization strictness")
    continuation_multiplier: float = Field(1.2, description="Penalty escalation multiplier")
    update_interval_epochs: int = Field(50, description="Epochs between dual updates")
    volume_tolerance: float = Field(0.02, description="Absolute volume tolerance limit")

class SystemConfig(BaseModel):
    physics: PhysicsConfig = PhysicsConfig()
    network: NetworkConfig = NetworkConfig()
    optim: OptimizationConfig = OptimizationConfig()

# Global Instantiation (Preserving backward compatibility with your scripts)
_config = SystemConfig()
PHYSICS = _config.physics
NETWORK = _config.network
OPTIM = _config.optim