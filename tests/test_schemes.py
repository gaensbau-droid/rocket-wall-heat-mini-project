import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from schemes import ftcs, crank_nicolson


def test_ftcs_stability():
    nx = 50
    L = 0.01
    dx = L / (nx - 1)
    alpha = 1.7e-5

    dt = (dx**2) / (2 * alpha) * 0.9
    nt = 10

    u0 = np.ones(nx) * 300
    u0[nx // 2] = 1000

    bc_left = 300
    bc_right = 300

    u = ftcs(u0.copy(), nx, dx, dt, nt, alpha, bc_left, bc_right)

    assert np.all(np.isfinite(u)), "FTCS produced NaNs or infinities"
    assert not np.any(u < 0), "FTCS produced negative temperatures"


def test_cn_conservation():
    nx = 50
    L = 0.01
    dx = L / (nx - 1)
    alpha = 1.7e-5

    dt = 0.1
    nt = 5

    u0 = np.ones(nx) * 300
    u0[nx // 2] = 1000

    bc_left = 300
    bc_right = 300

    u_cn = crank_nicolson(u0.copy(), nx, dx, dt, nt, alpha, bc_left, bc_right)

    assert np.isfinite(u_cn).all(), "CN produced NaNs"
    assert abs(u_cn.max() - 1000) < 5, "CN diffused too aggressively in few steps"
