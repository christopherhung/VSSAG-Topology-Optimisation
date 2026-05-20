import os
import time
import psutil
import torch
import numpy as np

class Profiler:
    """
    Rigorously tracks System RAM and GPU VRAM to mathematically prove 
    the CPU-Adjoint bypass claims in Chapter 4.
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = 0.0

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self):
        return time.time() - self.start_time

    def get_system_ram_gb(self):
        """
        Returns the Resident Set Size (RSS) RAM used by this Python process.
        Crucial for capturing the memory spike during SciPy's LU Factorization.
        """
        return self.process.memory_info().rss / (1024 ** 3)

    def get_peak_gpu_vram_mb(self):
        """
        Returns peak GPU VRAM used by PyTorch tensors in the current epoch.
        Resets the tracker immediately after calling to isolate epoch-by-epoch costs.
        """
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats() 
            return peak_vram
        return 0.0

def calculate_grey_volume(rho_phys_tensor, lower_bound=0.05, upper_bound=0.95):
    """
    Empirically quantifies the 'Acoustic Foam' Trap.
    Any pixel between 0.05 and 0.95 is deemed physically unmanufacturable.
    """
    # Detach from graph to prevent memory leaks during tracking
    rho = rho_phys_tensor.detach().cpu().numpy()
    total_elements = rho.size
    grey_elements = np.sum((rho > lower_bound) & (rho < upper_bound))
    return (grey_elements / total_elements) * 100.0

def calculate_mnd(rho_phys_tensor):
    """
    Measure of Non-Discreteness (M_nd). 
    A strict mathematical metric. M_nd = 0 means perfect 0/1 binarization.
    """
    rho = rho_phys_tensor.detach().cpu().numpy()
    mnd = np.sum(4 * rho * (1 - rho)) / rho.size
    return mnd