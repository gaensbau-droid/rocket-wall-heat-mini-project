"""
Spatial convergence study for the 1D heat equation solvers
(FTCS and Crank–Nicolson) defined in src.schemes.

using the same "bump diffusion" type initial condition as main.py,
but running it on multiple spatial grids and check how the L2 error
decreases as Δx → 0.
"""
import os
import sys

# Make sure the project root (the directory that contains `src/`) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

# import baseline configuration and material properties
from src.config import nx as NX_BASE, dx as DX_BASE, dt as DT_BASE, nt as NT_BASE, alpha
from src.schemes import ftcs, crank_nicolson

# infer physical domain length and final time from the config
L_DOMAIN = DX_BASE * (NX_BASE - 1)      # 0 .. L
T_FINAL = DT_BASE * NT_BASE            # total simulated time

# choose a target Courant/Fourier number below the FTCS limit
R_TARGET = 0.4                         # < 0.5 for stability


def run_solver(method: str, nx_local: int):
    """
    Run the chosen scheme (FTCS or CN) on a grid with nx_local points.

    Returns- 
    x : array, shape (nx_local,)
        spatial grid
    t : array, shape (nt_local + 1,)
        time grid
    u : array, shape (nt_local + 1, nx_local)
        temperature field
    """
    # spatial step for this grid, keep the same physical length
    dx_local = L_DOMAIN / (nx_local - 1)

    # pick dt based on desired r = alpha*dt/dx^2
    dt_local = R_TARGET * dx_local**2 / alpha

    # number of steps to reach approximately the same final time
    nt_local = int(np.round(T_FINAL / dt_local))
    # recompute dt so we land exactly on T_FINAL
    dt_local = T_FINAL / nt_local
    t = np.linspace(0.0, T_FINAL, nt_local + 1)

    # spatial grid
    x = np.linspace(0.0, L_DOMAIN, nx_local)

    # initial condition: 300 K everywhere, bump in the middle
    u0 = np.ones(nx_local) * 300.0
    u0[nx_local // 2] = 1000.0

    bc_left = 300.0
    bc_right = 300.0

    if method.upper() == "FTCS":
        u = ftcs(u0, nx_local, dx_local, dt_local, nt_local, alpha, bc_left, bc_right)
    elif method.upper() in ("CN", "CRANK-NICOLSON", "CRANK_NICOLSON"):
        u = crank_nicolson(
            u0, nx_local, dx_local, dt_local, nt_local, alpha, bc_left, bc_right
        )
    else:
        raise ValueError(f"Unknown method '{method}'")

    return x, t, u


def compute_l2_error(x_coarse, u_coarse_final, x_ref, u_ref_final):
    """
    L2 (RMS) error between coarse and reference solution at final time.
    Reference is interpolated onto the coarse grid.
    """
    u_ref_interp = np.interp(x_coarse, x_ref, u_ref_final)
    diff = u_coarse_final - u_ref_interp
    return np.sqrt(np.mean(diff**2))


def run_convergence(method: str, nx_list):
    """
    Run convergence study for a given method over a list of nx values.

    Uses the finest grid as reference.
    """
    method = method.upper()
    nx_list = sorted(nx_list)

    # reference = finest grid
    nx_ref = nx_list[-1]
    x_ref, t_ref, u_ref = run_solver(method, nx_ref)
    u_ref_final = u_ref[-1, :]

    dx_vals = []
    errors = []

    for nx_local in nx_list[:-1]:
        x, t, u = run_solver(method, nx_local)
        u_final = u[-1, :]
        dx_local = x[1] - x[0]

        err = compute_l2_error(x, u_final, x_ref, u_ref_final)
        dx_vals.append(dx_local)
        errors.append(err)

        print(
            f"[{method}] nx = {nx_local:4d}, "
            f"dx = {dx_local:.5e}, L2 error = {err:.5e}"
        )

    dx_vals = np.array(dx_vals)
    errors = np.array(errors)

    # estimate order of convergence from log–log slope
    if len(dx_vals) >= 2:
        p = np.polyfit(np.log(dx_vals), np.log(errors), 1)
        order = -p[0]
        print(f"\nEstimated order of convergence for {method}: {order:.3f}\n")
    else:
        print(f"\nNot enough grid levels to estimate order for {method}.\n")

    return dx_vals, errors


def plot_convergence(dx_ftcs, err_ftcs, dx_cn, err_cn):
    plt.figure(figsize=(7, 5))
    if dx_ftcs.size:
        plt.loglog(dx_ftcs, err_ftcs, "o-", label="FTCS")
    if dx_cn.size:
        plt.loglog(dx_cn, err_cn, "s-", label="Crank–Nicolson")

    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"L2 error at $t = T_{\mathrm{final}}$")
    plt.title("Spatial convergence: FTCS vs Crank–Nicolson")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/convergence_ftcs_cn.png", dpi=200)
    # plt.show()  # uncomment if you want it interactive


def main():
    # choose a set of grid sizes; last one will be the reference
    # you can tweak these if they blow up or are too small/large
    nx_list = [NX_BASE // 4, NX_BASE // 2, NX_BASE]

    print(f"Using base nx = {NX_BASE}, inferred L = {L_DOMAIN:.4e} m")
    print(f"Target T_final = {T_FINAL:.4e} s\n")

    print("=== FTCS convergence ===")
    dx_ftcs, err_ftcs = run_convergence("FTCS", nx_list)

    print("=== Crank–Nicolson convergence ===")
    dx_cn, err_cn = run_convergence("CN", nx_list)

    plot_convergence(dx_ftcs, err_ftcs, dx_cn, err_cn)


if __name__ == "__main__":
    main()
