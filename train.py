import os
import csv
import torch
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from typing import Callable, Tuple

from config import PHYSICS, OPTIM
from boundary_conditions import generate_cantilever_bcs, generate_mbb_half_bcs, generate_l_bracket_bcs
from graph_extractor import MeshToGraphExtractor
from physics_solver import AdjointCompliance
from utils_telemetry import Profiler, calculate_grey_volume

# Import the original VSSAG/Heaviside, but use the new StabilizedHyperSIREN
from networks import VSSAG_Extractor, HeavisideProjection
from networks_stabilized import StabilizedHyperSIREN

def save_topology_image(rho_tensor, nelx, nely, filename):
    """Renders the physical density matrix as a high-res academic image."""
    rho_numpy = rho_tensor.detach().cpu().numpy().reshape((nely, nelx))
    plt.figure(figsize=(6, 6 * (nely / nelx)))
    # 'gray_r' makes 1.0 (solid) black and 0.0 (void) white
    plt.imshow(rho_numpy, cmap='gray_r', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved final topology image: {filename}")

def execute_load_case(case_name: str, bc_func: Callable) -> None:
    print(f"\n========================================================")
    print(f" [{case_name}] Executing Hestenes-Powell ALM Loop")
    print(f"========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Domain and Graph
    freedofs, fixeddofs, F_csc, passive_mask = bc_func(PHYSICS.nelx, PHYSICS.nely)
    mask_tensor = torch.tensor(passive_mask, dtype=torch.float32, device=device)
    
    data = MeshToGraphExtractor(PHYSICS.nelx, PHYSICS.nely).extract(F_csc, fixeddofs).to(device)
    
    # 2. Initialize Stabilized Network Architecture
    vssag = VSSAG_Extractor().to(device)
    hyper_siren = StabilizedHyperSIREN(latent_dim=64, hidden_dim=128, rank=8).to(device)
    heaviside = HeavisideProjection(OPTIM.heaviside_initial_beta).to(device)
    
    trainable_params = list(vssag.parameters()) + list(hyper_siren.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=OPTIM.learning_rate, betas=OPTIM.adam_betas)
    
    # 3. Initialize ALM Parameters, Coordinates, and Baseline Tracker
    lambda_val = 0.0
    mu = 1.0  # ALM Penalty parameter reverted to stable baseline
    c_0 = None # Baseline compliance tracker for dynamic normalization
    
    coords = torch.stack(torch.meshgrid(torch.linspace(1, -1, PHYSICS.nely, device=device), 
                                        torch.linspace(-1, 1, PHYSICS.nelx, device=device), indexing='ij'), dim=-1).flatten(0,1)

    # 4. Initialize Empirical Telemetry Tracker
    profiler = Profiler()
    csv_filename = f'vssag_telemetry_{case_name}.csv'
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Compliance_C', 'Volume_Fraction_Vf', 'Grey_Volume_Pct', 
                         'ALM_Penalty_Mu', 'Heaviside_Beta', 'Peak_GPU_VRAM_MB', 
                         'Peak_System_RAM_GB', 'Epoch_Time_Sec'])

    # 5. Core ALM Training Loop
    for epoch in range(1, OPTIM.epochs + 1):
        profiler.start_timer()
        optimizer.zero_grad()
        
        # Forward Pass: Graph -> Stabilized HyperSIREN -> Heaviside
        rho_raw = hyper_siren(vssag(data.x_dict, data.edge_index_dict), coords).view(PHYSICS.nely, PHYSICS.nelx)
        rho_masked = rho_raw * (1 - mask_tensor) + 1e-3 * mask_tensor
        rho_phys = heaviside(rho_masked)
        
        # CPU Physics Bypass & Adjoint Sensitivity Injection
        compliance = AdjointCompliance.apply(rho_phys, freedofs, fixeddofs, F_csc, PHYSICS.nelx, PHYSICS.nely)
        
        # Dynamic Baseline Normalization
        if epoch == 1:
            c_0 = compliance.item()
            print(f"[*] Initial Raw Compliance (C_0) recorded: {c_0:.2f}")
            
        # 🚨 FINAL CALIBRATION: LINEAR NORMALISATION 🚨
        # We need linear gradients so the network can "heal" cracks.
        # Gradient clipping will handle the explosions.
        normalized_compliance = compliance / c_0
        
        # Calculate Volume Constraint Violation
        current_vf = torch.mean(rho_phys)
        h_v = current_vf - PHYSICS.vol_target
        
        # We use a gentle ALM_WEIGHT to balance the O(N) compliance sum against the O(1/N) volume mean.
        ALM_WEIGHT = 20.0 
        alm_penalty = ALM_WEIGHT * ((lambda_val * h_v) + (0.5 * mu * (h_v ** 2)))
        loss_al = normalized_compliance + alm_penalty
        
        loss_al.backward()
        
        # Strict gradient clipping to protect the SIREN manifold
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        
        # Hestenes-Powell Dual Update (Strict 2% Tolerance Gate)
        with torch.no_grad():
            lambda_val += mu * h_v.item()
            if epoch % OPTIM.update_interval_epochs == 0:
                if abs(h_v.item()) > OPTIM.volume_tolerance:
                    mu = min(mu * OPTIM.continuation_multiplier, OPTIM.alm_max_mu)
                heaviside.step_beta(OPTIM.continuation_multiplier, OPTIM.heaviside_max_beta)
                
        # 6. Harvest Telemetry Data
        epoch_time = profiler.end_timer()
        peak_vram = profiler.get_peak_gpu_vram_mb()
        sys_ram = profiler.get_system_ram_gb()
        grey_vol_pct = calculate_grey_volume(rho_phys)
        
        # Export Telemetry safely
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, compliance.item(), current_vf.item(), grey_vol_pct,
                             mu, heaviside.beta, peak_vram, sys_ram, epoch_time])
                
        # Console Reporting
        if epoch % 50 == 0 or epoch == 1:
            print(f"Ep {epoch:04d} | Raw C: {compliance.item():>8.2f} | Norm C: {normalized_compliance.item():>6.4f} | "
                  f"Vol: {current_vf.item():>5.4f} (err: {h_v.item():>+6.4f}) | "
                  f"Grey: {grey_vol_pct:>4.1f}% | \u03B2: {heaviside.beta:>4.1f}")
                  
        # Extract Final Image
        if epoch == OPTIM.epochs:
            save_topology_image(rho_phys, PHYSICS.nelx, PHYSICS.nely, f'Final_Topology_{case_name}.png')


if __name__ == "__main__":
    print("====================================================================")
    print(" VSSAG Topology Optimisation: Final Empirical Execution Suite")
    print("====================================================================")
    
    # Execute full suite
    execute_load_case("Cantilever", generate_cantilever_bcs)
    execute_load_case("MBB", generate_mbb_half_bcs)
    execute_load_case("L_Bracket", generate_l_bracket_bcs)
    
    print("\n[SUCCESS] Execution complete. Telemetry data and images exported.")