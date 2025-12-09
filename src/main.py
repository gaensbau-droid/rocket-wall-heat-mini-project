from .config import nx, dx, dt, nt, kappa, alpha, hg, Tg, pulse_is_on
from .grid import make_space_grid, make_time_grid

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

if __name__ == "__main__":
    main()
