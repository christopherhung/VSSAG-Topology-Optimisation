import numpy as np
import scipy.sparse as sps
from typing import Tuple

def _validate_mesh(nelx: int, nely: int) -> None:
    if not isinstance(nelx, int) or not isinstance(nely, int):
        raise TypeError("Mesh dimensions must be integers.")
    if nelx <= 0 or nely <= 0:
        raise ValueError("Mesh dimensions must be strictly positive.")

def generate_cantilever_bcs(nelx: int, nely: int) -> Tuple[np.ndarray, np.ndarray, sps.csc_matrix, np.ndarray]:
    """Standard Cantilever: Fixed left edge, downward load at bottom-right."""
    _validate_mesh(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = sps.lil_matrix((ndof, 1), dtype=np.float64)
    
    node_bottom_right = nelx * (nely + 1) + nely
    F[2 * node_bottom_right + 1, 0] = -1.0 

    fixeddofs = np.arange(0, 2 * (nely + 1))
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    passive_mask = np.zeros((nely, nelx), dtype=np.float32)
    return freedofs, fixeddofs, F.tocsc(), passive_mask

def generate_mbb_half_bcs(nelx: int, nely: int) -> Tuple[np.ndarray, np.ndarray, sps.csc_matrix, np.ndarray]:
    """Half-MBB: Symmetry on left edge, roller at bottom right, load at top-left."""
    _validate_mesh(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = sps.lil_matrix((ndof, 1), dtype=np.float64)
    
    node_top_left = 0
    F[2 * node_top_left + 1, 0] = -1.0

    fixed_left_x = np.arange(0, 2 * (nely + 1), 2)
    node_bottom_right = nelx * (nely + 1) + nely
    fixed_br_y = np.array([2 * node_bottom_right + 1])

    fixeddofs = np.union1d(fixed_left_x, fixed_br_y)
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    passive_mask = np.zeros((nely, nelx), dtype=np.float32)
    return freedofs, fixeddofs, F.tocsc(), passive_mask

def generate_l_bracket_bcs(nelx: int, nely: int) -> Tuple[np.ndarray, np.ndarray, sps.csc_matrix, np.ndarray]:
    """L-Bracket: Top-right void, fixed top edge (left side), load at bottom-right."""
    _validate_mesh(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = sps.lil_matrix((ndof, 1), dtype=np.float64)
    
    node_bottom_right = nelx * (nely + 1) + nely
    F[2 * node_bottom_right + 1, 0] = -1.0

    fix_x_limit = int(nelx * 0.4)
    fixed_nodes = [x * (nely + 1) + 0 for x in range(fix_x_limit + 1)]
    
    fixeddofs_list = []
    for n in fixed_nodes:
        fixeddofs_list.extend([2 * n, 2 * n + 1])
    fixeddofs = np.array(fixeddofs_list, dtype=int)
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    passive_mask = np.zeros((nely, nelx), dtype=np.float32)
    for x in range(fix_x_limit, nelx):
        for y in range(0, int(nely * 0.6)):
            passive_mask[y, x] = 1.0 
    return freedofs, fixeddofs, F.tocsc(), passive_mask