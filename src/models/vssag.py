"""
vssag.py
Variational State-Space Augmented Graph (VSSAG) transport layer.
Executes non-dissipative continuous-time ODE via RK4 integration.
"""
import torch
import torch.nn as nn
from config import NETWORK

class VSSAGLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Ensure dimensions match for the anti-symmetric constraint
        assert in_features == out_features, "Input and output features must be equal for anti-symmetry."
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W)
        self.dt = NETWORK.vssag_dt
        self.steps = NETWORK.vssag_layers_L

    def _anti_symmetric_weight(self) -> torch.Tensor:
        """
        Enforces mathematically non-dissipative physical load transfer 
        to prevent topological over-squashing.
        """
        return self.W - self.W.T

    def _f(self, H: torch.Tensor, M_ne: torch.Tensor) -> torch.Tensor:
        """
        Computes the state derivative.
        M_ne is the aggregated message tensor from neighboring nodes.
        """
        W_tilde = self._anti_symmetric_weight()
        return torch.tanh(torch.matmul(M_ne, W_tilde))

    def forward(self, H_e: torch.Tensor, M_ne: torch.Tensor) -> torch.Tensor:
        """
        4th-Order Runge-Kutta (RK4) Integration over L pseudo-time steps.
        """
        H = H_e
        for _ in range(self.steps):
            k1 = self._f(H, M_ne)
            k2 = self._f(H + (self.dt / 2) * k1, M_ne)
            k3 = self._f(H + (self.dt / 2) * k2, M_ne)
            k4 = self._f(H + self.dt * k3, M_ne)
            
            H = H + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
        return H