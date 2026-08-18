"""
main.py
Primary execution orchestrator for the VSSAG framework.
Wires the continuous neural manifold to the discrete CPU physics solver.
"""
import torch
import argparse
from config import config, PHYSICS, OPTIM
from models import VSSAGLayer, FactorisedHypernetwork, SIREN
from utils import DynamicHeaviside, ALMController
from physics import AdjointCompliance

def train():
    # Hardware allocation (Physics remains on CPU, Neural on GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing VSSAG Pipeline on {device}...")

    # Initialize Neural Components
    vssag = VSSAGLayer(in_features=64, out_features=64).to(device)
    hypernet = FactorisedHypernetwork().to(device)
    siren = SIREN().to(device)
    
    # Initialize Constraint Controllers
    heaviside = DynamicHeaviside()
    alm = ALMController()
    
    # Optimizer (Primal Step)
    optimizer = torch.optim.Adam(
        list(vssag.parameters()) + list(hypernet.parameters()) + list(siren.parameters()),
        lr=OPTIM.learning_rate,
        betas=OPTIM.adam_betas
    )

    print("Commencing ALM Optimization Loop...")
    for epoch in range(1, OPTIM.epochs + 1):
        optimizer.zero_grad()
        
        # 1. Forward Pass: Neural Manifold (Mock tensors for structural wiring)
        # In a full run, these are populated by the bipartite graph extraction
        dummy_h_e = torch.randn(10000, 64, device=device) 
        dummy_m_ne = torch.randn(10000, 64, device=device)
        dummy_coords = torch.randn(10000, 2, device=device)
        
        # VSSAG Transport -> Hypernet -> SIREN Density
        h_transported = vssag(dummy_h_e, dummy_m_ne)
        h_global = torch.mean(h_transported, dim=0, keepdim=True)
        w_siren = hypernet(h_global)
        rho_tilde = siren(dummy_coords, w_siren)
        
        # 2. Strict Binarization Projection
        rho_phys = heaviside.forward(rho_tilde)
        
        # 3. Exact Adjoint Physics Bypass (Mock sparse matrices for wiring)
        # AdjointCompliance.apply(...) would execute here, returning scalar compliance.
        dummy_compliance = torch.tensor(100.0, requires_grad=True, device=device)
        
        # 4. ALM Saddle-Point Loss
        loss = alm.compute_loss(dummy_compliance, rho_phys)
        loss.backward()
        optimizer.step()
        
        # 5. Dual Step: ALM Penalty & Heaviside Update
        if epoch % OPTIM.update_interval_epochs == 0:
            alm.dual_update(rho_phys, heaviside)
            print(f"Epoch {epoch} | ALM Dual Update Executed | Beta: {heaviside.beta:.2f} | Mu: {alm.mu:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSSAG Topology Optimization")
    parser.add_argument("--config-profile", type=str, default="cantilever_100", help="Configuration profile to execute")
    args = parser.parse_args()
    
    train()