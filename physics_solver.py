"""
physics_solver.py
Executes the exact discrete adjoint bypass to achieve a 0.0 MB GPU physics footprint.
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import torch
from typing import Any, Tuple
from config import PHYSICS
from utils_telemetry import Profiler

# Initialise singleton profiler for the physics pass telemetry
physics_profiler = Profiler()

def solve_kinematics_cpu(K_sparse, F_vector):
    """
    Executes the direct LU factorisation.
    
    NOTE: Iterative solvers (e.g., Conjugate Gradient, MINRES) are 
    explicitly bypassed here. The void regions in topology optimisation (E_min) 
    create highly ill-conditioned matrices that induce severe iterative instability. 
    By utilising `spsolve` (wrapping SuperLU/UMFPACK), guarantee exact direct 
    mathematical equilibrium bounded only by IEEE 754 machine precision.
    """
    physics_profiler.start_timer()
    
    # Execute standard SciPy backend
    # This matrix factorisation happens entirely in System RAM, bypassing the GPU.
    U = spsolve(K_sparse, F_vector)
    
    solve_time_sec = physics_profiler.end_timer()
    peak_system_ram_gb = physics_profiler.get_system_ram_gb()
    
    return U, solve_time_sec, peak_system_ram_gb

class AdjointCompliance(torch.autograd.Function):
    """
    Executes the analytical CPU-Adjoint bypass. Solves KU=F using SuperLU/UMFPACK 
    and analytically injects the exact discrete gradient back into PyTorch.
    """
    @staticmethod
    def forward(ctx: Any, rho_phys: torch.Tensor, freedofs: np.ndarray, fixeddofs: np.ndarray, F_csc: sps.csc_matrix) -> torch.Tensor:
        # Move tensor to CPU and enforce singularity guard (E_min boundary)
        rho_np = np.maximum(rho_phys.detach().cpu().numpy().flatten('F'), 1e-3)
        ndof = 2 * (PHYSICS.nelx + 1) * (PHYSICS.nely + 1)
        
        # 1. Base Element Stiffness Matrix (KE) Assembly for 2D Bilinear Quad Elements
        k = np.array([1/2-PHYSICS.nu/6, 1/8+PHYSICS.nu/8, -1/4-PHYSICS.nu/12, -1/8+3*PHYSICS.nu/8,
                      -1/4+PHYSICS.nu/12, -1/8-PHYSICS.nu/8, PHYSICS.nu/6, 1/8-3*PHYSICS.nu/8])
        KE = PHYSICS.E0 / (1 - PHYSICS.nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])

        # 2. Vectorised DOF Mapping (Avoiding O(N) Python loops)
        nodenrs = np.arange(1, 1 + (PHYSICS.nelx+1)*(PHYSICS.nely+1)).reshape(PHYSICS.nelx+1, PHYSICS.nely+1).T
        edofVec = nodenrs[0:PHYSICS.nely, 0:PHYSICS.nelx].flatten('F') * 2
        edofVec = edofVec.reshape((PHYSICS.nelx*PHYSICS.nely, 1))
        edofMat = np.tile(edofVec, (1, 8)) + np.tile(
            np.array([0, 1, 2*PHYSICS.nely+2, 2*PHYSICS.nely+3, 2*PHYSICS.nely, 2*PHYSICS.nely+1, -2, -1]), (PHYSICS.nelx*PHYSICS.nely, 1)
        )
        iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 8))).flatten()

        # 3. Direct Sparse Assembly & Kinematic Solve (CPU Bound)
        E_rho = PHYSICS.Emin + (rho_np ** PHYSICS.penal) * (PHYSICS.E0 - PHYSICS.Emin)
        sK = (E_rho[:, np.newaxis] * KE.flatten()).flatten()
        K = sps.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

        U = np.zeros(ndof)
        # Slices the global matrix to solve only for the free degrees of freedom
        U[freedofs] = spsolve(K[freedofs, :][:, freedofs], F_csc[freedofs, :])

        # 4. Local Element Strain Energy (ce) & Global Compliance Calculation
        U_e = U[edofMat]
        ce = np.sum(np.dot(U_e, KE) * U_e, axis=1) # ce = u_e^T * k0 * u_e
        compliance = float(np.sum(E_rho * ce))

        # CACHE FOR BACKWARD: Store ONLY the scalar strains and densities.
        # This reduces the PyTorch Autograd memory footprint to an absolute minimum.
        ctx.save_for_backward(torch.tensor(rho_np, device=rho_phys.device), torch.tensor(ce, device=rho_phys.device))
        ctx.shape = rho_phys.shape

        return torch.tensor(compliance, requires_grad=True, device=rho_phys.device)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        The Backward Pass: Analytical Adjoint Injection.
        Executes Equation: dC/drho = -p * rho^(p-1) * (E0 - Emin) * u^T * k0 * u
        """
        rho_np_tensor, ce_tensor = ctx.saved_tensors
        
        # Exact Analytical Sensitivity calculated entirely via O(1) element-wise operations
        grad_rho = -PHYSICS.penal * (rho_np_tensor ** (PHYSICS.penal - 1)) * (PHYSICS.E0 - PHYSICS.Emin) * ce_tensor
        
        # Chain Rule Injection: Route gradient back into the SIREN neural weights
        return grad_output * grad_rho.view(ctx.shape), None, None, None