import os
import torch
import numpy as np
import scipy.sparse as sps

from config import PHYSICS, OPTIM
from boundary_conditions import generate_cantilever_bcs
from graph_extractor import MeshToGraphExtractor
from physics_solver import AdjointCompliance
from utils_telemetry import calculate_grey_volume
from networks import VSSAG_Extractor, HeavisideProjection
from networks_stabilized import StabilizedHyperSIREN

def set_random_seed(seed):
    """Locks all stochastic generators to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_single_seed(seed, run_id):
    set_random_seed(seed)
    print(f"\n[Robustness Test] Executing Run {run_id}/5 (Seed: {seed})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freedofs, fixeddofs, F_csc, passive_mask = generate_cantilever_bcs(PHYSICS.nelx, PHYSICS.nely)
    mask_tensor = torch.tensor(passive_mask, dtype=torch.float32, device=device)
    data = MeshToGraphExtractor(PHYSICS.nelx, PHYSICS.nely).extract(F_csc, fixeddofs).to(device)
    
    vssag = VSSAG_Extractor().to(device)
    hyper_siren = StabilizedHyperSIREN(latent_dim=64, hidden_dim=128, rank=8).to(device)
    
    # 🚨 ISOLATION OVERRIDE: Dampen frequency to isolate ALM stability 🚨
    hyper_siren.omega_0 = 5.0 
    
    heaviside = HeavisideProjection(OPTIM.heaviside_initial_beta).to(device)
    
    trainable_params = list(vssag.parameters()) + list(hyper_siren.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=OPTIM.learning_rate, betas=OPTIM.adam_betas)
    
    lambda_val, mu = 0.0, 1.0
    c_0 = None
    coords = torch.stack(torch.meshgrid(torch.linspace(1, -1, PHYSICS.nely, device=device), 
                                        torch.linspace(-1, 1, PHYSICS.nelx, device=device), indexing='ij'), dim=-1).flatten(0,1)

    TEST_EPOCHS = 1500 # Must match full training run for accurate variance
    
    for epoch in range(1, TEST_EPOCHS + 1):
        optimizer.zero_grad()
        rho_raw = hyper_siren(vssag(data.x_dict, data.edge_index_dict), coords).view(PHYSICS.nely, PHYSICS.nelx)
        rho_masked = rho_raw * (1 - mask_tensor) + 1e-3 * mask_tensor
        rho_phys = heaviside(rho_masked)
        
        compliance = AdjointCompliance.apply(rho_phys, freedofs, fixeddofs, F_csc)
        if epoch == 1: c_0 = compliance.item()
            
        normalized_compliance = compliance / c_0
        current_vf = torch.mean(rho_phys)
        h_v = current_vf - PHYSICS.vol_target
        
        # 🚨 ISOLATION OVERRIDE: Tightened ALM Weight for absolute rigid bounding 🚨
        ALM_WEIGHT = 50.0 
        loss_al = normalized_compliance + (ALM_WEIGHT * ((lambda_val * h_v) + (0.5 * mu * (h_v ** 2))))
        loss_al.backward()
        
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            lambda_val += mu * h_v.item()
            if epoch % OPTIM.update_interval_epochs == 0:
                if abs(h_v.item()) > OPTIM.volume_tolerance:
                    mu = min(mu * OPTIM.continuation_multiplier, OPTIM.alm_max_mu)
                heaviside.step_beta(OPTIM.continuation_multiplier, OPTIM.heaviside_max_beta)
                
    final_c = compliance.item()
    final_v = current_vf.item()
    final_grey = calculate_grey_volume(rho_phys)
    print(f"-> Run {run_id} Completed | C: {final_c:.2f} | Vf: {final_v:.4f} | Grey: {final_grey:.2f}%")
    
    return final_c, final_v, final_grey

if __name__ == "__main__":
    seeds = [42, 1024, 777, 2026, 8] 
    results = []
    
    for i, seed in enumerate(seeds):
        results.append(run_single_seed(seed, i+1))
        
    c_arr = np.array([r[0] for r in results])
    v_arr = np.array([r[1] for r in results])
    g_arr = np.array([r[2] for r in results])
    
    print("\n========================================================")
    print(" STOCHASTIC UNCERTAINTY ANALYSIS RESULTS (N=5 runs)")
    print("========================================================")
    print(f"Compliance (C):      {c_arr.mean():.2f} +/- {c_arr.std():.2f}")
    print(f"Volume Fraction (Vf): {v_arr.mean():.4f} +/- {v_arr.std():.4f}")
    print(f"Grey Volume (%):     {g_arr.mean():.2f}% +/- {g_arr.std():.2f}%")
    print("========================================================")