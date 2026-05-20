import torch
import torch.nn as nn

class StabilizedHyperSIREN(nn.Module):
    """
    A ruthlessly stabilized Hypernetwork + SIREN block.
    Engineered to prevent Hypernetwork Variance Explosion and SIREN Spectral Collapse.
    """
    def __init__(self, latent_dim=64, hidden_dim=128, rank=8, out_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Hypernetwork for Layer 1 (Input 2 -> Hidden 128)
        self.fc_A1 = nn.Linear(latent_dim, hidden_dim * rank)
        self.fc_B1 = nn.Linear(latent_dim, rank * 2)
        self.bias_1 = nn.Linear(latent_dim, hidden_dim)
        
        # 2. Hypernetwork for Layer 2 (Hidden 128 -> Output 1)
        self.fc_A2 = nn.Linear(latent_dim, out_dim * rank)
        self.fc_B2 = nn.Linear(latent_dim, rank * hidden_dim)
        self.bias_2 = nn.Linear(latent_dim, out_dim)

        # 🚨 THE MATHEMATICAL ARMOR: STRICT VARIANCE SCALING 🚨
        # We must shrink the initial weights so A x B doesn't explode.
        for m in [self.fc_A1, self.fc_B1, self.bias_1, self.fc_A2, self.fc_B2, self.bias_2]:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)
            # 🚨 FIX 2: FORCE SOLID INITIALIZATION 🚨
            # A massive positive bias forces the final Sigmoid to output ~0.99 (Solid) on Epoch 1
            nn.init.constant_(self.bias_2.bias, 0.0)
            
        # Force low spatial frequency
        self.omega_0 = 5.0

    def forward(self, h_global, coords):
        """
        h_global: [1, 64] latent vector from VSSAG
        coords: [10000, 2] spatial coordinates
        """
        # Ensure h_global is 2D
        if h_global.dim() == 1:
            h_global = h_global.unsqueeze(0)
            
        # --- GENERATE WEIGHTS ---
        # Layer 1
        A1 = self.fc_A1(h_global).view(self.hidden_dim, -1) # [128, 8]
        B1 = self.fc_B1(h_global).view(-1, 2)               # [8, 2]
        W1 = torch.matmul(A1, B1) * 0.1                     # [128, 2] (SCALED!)
        b1 = self.bias_1(h_global).view(self.hidden_dim)    # [128]
        
        # Layer 2
        A2 = self.fc_A2(h_global).view(1, -1)               # [1, 8]
        B2 = self.fc_B2(h_global).view(-1, self.hidden_dim) # [8, 128]
        W2 = torch.matmul(A2, B2) * 0.1                     # [1, 128] (SCALED!)
        b2 = self.bias_2(h_global).view(1)                  # [1]

        # --- EXECUTE SIREN ---
        # Layer 1: Sine Activation
        # coords is [10000, 2]. W1 is [128, 2]. 
        # linear output: (coords @ W1^T) + b1 -> [10000, 128]
        hidden = torch.sin(self.omega_0 * (torch.matmul(coords, W1.t()) + b1))
        
        # Layer 2: Output with Sigmoid mapping to (0, 1)
        # hidden is [10000, 128]. W2 is [1, 128].
        out = torch.sigmoid(torch.matmul(hidden, W2.t()) + b2) # [10000, 1]
        
        return out