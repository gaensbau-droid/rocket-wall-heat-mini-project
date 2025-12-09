import numpy as np
from .config import L, nx, dt, nt

def make_space_grid():
    x = np.linspace(0.0, L, nx)
    return x

def make_time_grid():
    t_final = nt * dt
    t = np.linspace(0.0, t_final, nt + 1)
    return t
