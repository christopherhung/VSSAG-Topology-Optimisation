"""
constraints.py
Enforces strict physical manufacturability via dynamic Heaviside projection 
and Augmented Lagrangian (ALM) saddle-point optimization.
"""
import torch
from config import OPTIM, PHYSICS

class DynamicHeaviside:
    """
    Projects continuous neural density into strict binary physical states.
    Escalates steepness (beta) to eradicate 'acoustic foam' intermediate densities.
    """
    def __init__(self):
        self.beta = OPTIM.heaviside_initial_beta
        self.max_beta = OPTIM.heaviside_max_beta
        self.multiplier = OPTIM.continuation_multiplier

    def forward(self, rho_tilde: torch.Tensor) -> torch.Tensor:
        """
        Applies differentiable continuation-based Heaviside step function.
        """
        thresh = 0.5
        num = torch.tanh(self.beta * thresh) + torch.tanh(self.beta * (rho_tilde - thresh))
        den = torch.tanh(self.beta * thresh) + torch.tanh(self.beta * (1.0 - thresh))
        return num / den

    def step(self):
        """Escalates strictness gate."""
        self.beta = min(self.beta * self.multiplier, self.max_beta)

class ALMController:
    """
    Primal-Dual Hestenes-Powell Augmented Lagrangian constraint enforcer.
    """
    def __init__(self):
        self.lam = 0.0  # First-Order Lagrangian Multiplier
        self.mu = OPTIM.alm_initial_mu  # Quadratic Penalty
        self.max_mu = OPTIM.alm_max_mu
        self.multiplier = OPTIM.continuation_multiplier
        self.tolerance = OPTIM.volume_tolerance

    def compute_violation(self, rho_phys: torch.Tensor) -> torch.Tensor:
        """Calculates current volume fraction violation h_v(rho)."""
        current_vol = torch.mean(rho_phys)
        return current_vol - PHYSICS.vol_target

    def compute_loss(self, compliance: torch.Tensor, rho_phys: torch.Tensor) -> torch.Tensor:
        """
        Computes the unconstrained saddle-point objective.
        """
        h_v = self.compute_violation(rho_phys)
        return compliance + (self.lam * h_v) + (0.5 * self.mu * (h_v ** 2))

    def dual_update(self, rho_phys: torch.Tensor, heaviside_proj: DynamicHeaviside):
        """
        Executes the Dual Step (Multiplier & Penalty Update) conditionally.
        If the absolute volume constraint violation exceeds the tolerance threshold,
        both mu and beta are escalated to override the 'cheating' optimizer.
        """
        h_v = self.compute_violation(rho_phys).item()
        
        # First-Order Lagrange Multiplier Update
        self.lam = self.lam + self.mu * h_v
        
        # Conditional Strictness Gate
        if abs(h_v) > self.tolerance:
            self.mu = min(self.mu * self.multiplier, self.max_mu)
            heaviside_proj.step()