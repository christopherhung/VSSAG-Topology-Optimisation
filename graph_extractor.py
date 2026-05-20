import numpy as np
import scipy.sparse as sps
import torch
from torch_geometric.data import HeteroData
from typing import List

class MeshToGraphExtractor:
    """Translates the P0/P1 FEA domain into a PyTorch Geometric Bipartite Graph."""
    def __init__(self, nelx: int, nely: int) -> None:
        self.nelx = nelx
        self.nely = nely
        self.num_nodes = (nelx + 1) * (nely + 1)
        self.num_elements = nelx * nely
        self.ndof = 2 * self.num_nodes

    def extract(self, F: sps.csc_matrix, fixeddofs: np.ndarray) -> HeteroData:
        data = HeteroData()

        # 1. P1 Kinematic Node Features
        X_nodes = np.zeros((self.num_nodes, 6), dtype=np.float32)
        x_coords, y_coords = np.meshgrid(np.linspace(0, 1, self.nelx + 1), np.linspace(1, 0, self.nely + 1))
        X_nodes[:, 0], X_nodes[:, 1] = x_coords.flatten('F'), y_coords.flatten('F')
        
        dir_mask = np.zeros(self.ndof, dtype=np.float32)
        dir_mask[fixeddofs] = 1.0
        X_nodes[:, 2], X_nodes[:, 3] = dir_mask[0::2], dir_mask[1::2]
        
        F_dense = F.toarray().flatten()
        X_nodes[:, 4], X_nodes[:, 5] = F_dense[0::2], F_dense[1::2]
        data['node'].x = torch.tensor(X_nodes, dtype=torch.float32)

        # 2. P0 Thermodynamic Element Features
        X_elements = np.zeros((self.num_elements, 2), dtype=np.float32)
        ex_coords, ey_coords = np.meshgrid(
            np.linspace(0.5 / self.nelx, 1 - 0.5 / self.nelx, self.nelx),
            np.linspace(1 - 0.5 / self.nely, 0.5 / self.nely, self.nely)
        )
        X_elements[:, 0], X_elements[:, 1] = ex_coords.flatten('F'), ey_coords.flatten('F')
        data['element'].x = torch.tensor(X_elements, dtype=torch.float32)

        # 3. Bipartite Geometry Generation
        edges_n2e, edges_e2n, attr_n2e, attr_e2n = [], [], [], []
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el_idx = elx * self.nely + ely
                nodes = [
                    ely + elx * (self.nely + 1),               
                    ely + 1 + elx * (self.nely + 1),           
                    ely + 1 + (elx + 1) * (self.nely + 1),     
                    ely + (elx + 1) * (self.nely + 1)          
                ]
                cx, cy = float(X_elements[el_idx, 0]), float(X_elements[el_idx, 1])
                for n in nodes:
                    nx, ny = float(X_nodes[n, 0]), float(X_nodes[n, 1])
                    dx, dy = cx - nx, cy - ny
                    edges_n2e.append([n, el_idx]); attr_n2e.append([dx, dy])
                    edges_e2n.append([el_idx, n]); attr_e2n.append([-dx, -dy])

        data['node', 'connects_to', 'element'].edge_index = torch.tensor(edges_n2e, dtype=torch.long).t().contiguous()
        data['node', 'connects_to', 'element'].edge_attr = torch.tensor(attr_n2e, dtype=torch.float32)
        data['element', 'connects_to', 'node'].edge_index = torch.tensor(edges_e2n, dtype=torch.long).t().contiguous()
        data['element', 'connects_to', 'node'].edge_attr = torch.tensor(attr_e2n, dtype=torch.float32)

        return data