import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Dict, Tuple, List
from config import NETWORK

class HeavisideProjection(nn.Module):
    def __init__(self, initial_beta: float, eta: float = 0.5):
        super().__init__()
        self.beta = initial_beta
        self.eta = eta

    def forward(self, rho_raw: torch.Tensor) -> torch.Tensor:
        beta_t = torch.tensor(self.beta, dtype=rho_raw.dtype, device=rho_raw.device)
        eta_t = torch.tensor(self.eta, dtype=rho_raw.dtype, device=rho_raw.device)
        numerator = torch.tanh(beta_t * eta_t) + torch.tanh(beta_t * (rho_raw - eta_t))
        denominator = torch.tanh(beta_t * eta_t) + torch.tanh(beta_t * (1.0 - eta_t))
        return numerator / denominator
    
    def step_beta(self, multiplier: float, max_beta: float) -> None:
        self.beta = min(self.beta * multiplier, max_beta)

class VSSAGMessage(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')
    def forward(self, x_src, edge_index, bipartite_size):
        return self.propagate(edge_index, x=x_src, size=bipartite_size)
    def message(self, x_j):
        return x_j

class VSSAGLayer_RK4(nn.Module):
    def __init__(self, hidden_dim: int, edge_types: List[Tuple[str, str, str]], dt: float):
        super().__init__()
        self.dt = dt
        self.W_dicts = nn.ParameterDict({
            f"{rel[0]}_{rel[1]}_{rel[2]}": nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)) for rel in edge_types
        })
        self.mp = VSSAGMessage()
        for param in self.W_dicts.values():
            nn.init.orthogonal_(param)

    def _ode_dynamics(self, x_dict, edge_index_dict):
        out_dict = {k: torch.zeros_like(v) for k, v in x_dict.items()}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel, dst_type = edge_type
            W = self.W_dicts[f"{src_type}_{rel}_{dst_type}"]
            W_anti = W - W.t() # Strict anti-symmetry constraint
            m_dst = self.mp(x_dict[src_type], edge_index, bipartite_size=(x_dict[src_type].size(0), x_dict[dst_type].size(0)))
            out_dict[dst_type] += F.linear(m_dst, W_anti)
        return {k: torch.tanh(v) for k, v in out_dict.items()}

    def forward(self, x_dict, edge_index_dict):
        k1 = self._ode_dynamics(x_dict, edge_index_dict)
        h_k2 = {k: x_dict[k] + (self.dt / 2.0) * k1[k] for k in x_dict.keys()}
        k2 = self._ode_dynamics(h_k2, edge_index_dict)
        h_k3 = {k: x_dict[k] + (self.dt / 2.0) * k2[k] for k in x_dict.keys()}
        k3 = self._ode_dynamics(h_k3, edge_index_dict)
        h_k4 = {k: x_dict[k] + self.dt * k3[k] for k in x_dict.keys()}
        k4 = self._ode_dynamics(h_k4, edge_index_dict)
        return {k: x_dict[k] + (self.dt / 6.0) * (k1[k] + 2*k2[k] + 2*k3[k] + k4[k]) for k in x_dict.keys()}

class VSSAG_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_proj = nn.Linear(6, NETWORK.latent_dim_D)
        self.elem_proj = nn.Linear(2, NETWORK.latent_dim_D)
        self.edge_types = [('node', 'connects_to', 'element'), ('element', 'connects_to', 'node')]
        self.layers = nn.ModuleList([
            VSSAGLayer_RK4(NETWORK.latent_dim_D, self.edge_types, dt=NETWORK.vssag_dt) for _ in range(NETWORK.vssag_layers_L)
        ])

    def forward(self, x_dict, edge_index_dict):
        h_dict = {'node': self.node_proj(x_dict['node']), 'element': self.elem_proj(x_dict['element'])}
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
        return h_dict['element'].mean(dim=0, keepdim=True) 

class FactorizedHypernetwork(nn.Module):
    def __init__(self):
        super().__init__()
        D, H, R = NETWORK.latent_dim_D, NETWORK.siren_hidden_H, NETWORK.hypernet_rank_R
        self.w1_net = nn.Linear(D, 2 * H); self.b1_net = nn.Linear(D, H)
        self.w2_A_net = nn.Linear(D, H * R); self.w2_B_net = nn.Linear(D, R * H)
        self.b2_net = nn.Linear(D, H)
        self.w3_net = nn.Linear(D, H * 1); self.b3_net = nn.Linear(D, 1)

    def forward(self, h_global):
        H, R = NETWORK.siren_hidden_H, NETWORK.hypernet_rank_R
        return {
            'w1': self.w1_net(h_global).view(H, 2), 'b1': self.b1_net(h_global).view(H),
            'w2': torch.matmul(self.w2_A_net(h_global).view(H, R), self.w2_B_net(h_global).view(R, H)),
            'b2': self.b2_net(h_global).view(H),
            'w3': self.w3_net(h_global).view(1, H), 'b3': self.b3_net(h_global).view(1)
        }

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega_0 = NETWORK.siren_omega_0
    def forward(self, weights, coords):
        x = torch.sin(self.omega_0 * F.linear(coords, weights['w1'], weights['b1']))
        x = torch.sin(self.omega_0 * F.linear(x, weights['w2'], weights['b2']))
        return torch.sigmoid(F.linear(x, weights['w3'], weights['b3']))