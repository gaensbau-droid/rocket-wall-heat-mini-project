from .config import nx, dx, dt, nt, kappa, alpha, hg, Tg, pulse_is_on
from .grid import make_space_grid, make_time_grid
from .schemes import ftcs
import numpy as np

def main():
    print("Environment OK.")
    print(f"nx = {nx}, dx = {dx:.6e}")
    print(f"dt = {dt}, nt = {nt}")
    print(f"alpha = {alpha}, kappa = {kappa}")

    x = make_space_grid()
    t = make_time_grid()

    print(f"Space grid: x[0]={x[0]:.6e}, x[-1]={x[-1]:.6e}, len={len(x)}")
    print(f"Time grid: t[0]={t[0]:.6e}, t[-1]={t[-1]:.6e}, len={len(t)}")

    print(f"Pulse at t=0.05s? {pulse_is_on(0.05)}")
    print(f"Pulse at t=0.19s? {pulse_is_on(0.19)}")
    print(f"Pulse at t=0.21s? {pulse_is_on(0.21)}")

    # FTCS Test
    print("\nRunning FTCS test...")

    # initial temperature array: uniform + bump
    u0 = np.ones(nx) * 300.0
    u0[nx // 2] = 1000.0

    bc_left = 300.0
    bc_right = 300.0

    n_steps_test = 200

    u = ftcs(u0, nx, dx, dt, n_steps_test, alpha, bc_left, bc_right)

    print("FTCS Okay: shape =", u.shape)
    print("Center temp t=0:", u[0, nx // 2])
    print("Center temp t=end:", u[-1, nx // 2])

if __name__ == "__main__":
    main()
