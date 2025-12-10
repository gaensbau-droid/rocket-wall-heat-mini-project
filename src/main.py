from .config import nx, dx, dt, nt, kappa, alpha, hg, Tg, pulse_is_on
from .grid import make_space_grid, make_time_grid
from .schemes import ftcs, heat_wall_ftcs, crank_nicolson
import numpy as np
import matplotlib.pyplot as plt

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

    # Physical rocket-wall FTCS simulation
    print("\nRunning physical rocket-wall FTCS simulation...")

    # initial wall temp: 300 K everywhere
    T0 = np.full(nx, 300.0)

    T_wall = heat_wall_ftcs(
        T0,
        nx, dx, dt, nt,
        alpha, kappa, hg, Tg,
        pulse_is_on,
    )

    print("Simulation complete. Field shape:", T_wall.shape)
    print("Hot-side wall temp at t=0:   ", T_wall[0, 0])
    print("Hot-side wall temp at t=end: ", T_wall[-1, 0])
    print("Cold-side wall temp at t=end:", T_wall[-1, -1])

    # plot 1 - temp vs x at several times
    times_to_plot = [0.0, 0.1, 0.3, 1.0, 1.5]

    plt.figure(figsize=(8, 5))
    for t_target in times_to_plot:
        idx = np.argmin(np.abs(t - t_target))
        plt.plot(x, T_wall[idx, :], label=f"t={t_target:.2f} s")
        plt.xlabel("x (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Wall temperature profiles at selected times (FTCS)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/wall_profiles_ftcs.png", dpi=200)
    # plt.show()

    # Plot 2: contour-style plot of T(x, t)
    plt.figure(figsize=(8, 5))
    # t is length nt+1, x is length nx; T_wall has shape (nt+1, nx)
    pcm = plt.pcolormesh(x, t, T_wall, shading="auto")
    plt.xlabel("x (m)")
    plt.ylabel("t (s)")
    plt.title("Rocket wall temperature field (FTCS)")
    cbar = plt.colorbar(pcm)
    cbar.set_label("Temperature (K)")
    plt.tight_layout()
    plt.savefig("figures/wall_contour_ftcs.png", dpi=200)
    # plt.show()

    

    # Simple FTCS vs CN diffusion test (internal consistency)
    print("\nRunning FTCS + CN internal diffusion test...")

    u0 = np.ones(nx) * 300.0
    u0[nx // 2] = 1000.0  # bump in the middle

    bc_left = 300.0
    bc_right = 300.0
    n_steps_test = 200

    # FTCS
    u_ftcs = ftcs(u0, nx, dx, dt, n_steps_test, alpha, bc_left, bc_right)

    # Crank–Nicolson
    u_cn = crank_nicolson(u0, nx, dx, dt, n_steps_test, alpha, bc_left, bc_right)

    print("FTCS shape:", u_ftcs.shape, "CN shape:", u_cn.shape)
    print("FTCS center temp t=0:   ", u_ftcs[0, nx // 2])
    print("FTCS center temp t=end: ", u_ftcs[-1, nx // 2])
    print("CN   center temp t=0:   ", u_cn[0, nx // 2])
    print("CN   center temp t=end: ", u_cn[-1, nx // 2])

    # quant comparison between ftcs and cn
    diff_max = np.max(np.abs(u_ftcs - u_cn))
    diff_rms = np.sqrt(np.mean((u_ftcs - u_cn) **2))

    print(f"Max |FTCS - CN| over doman and time: {diff_max: .6e} K")
    print(f"RMS |FTCS - CN| over domain and time: {diff_rms:.6e} K")

    # comparison plot
    plt.figure(figsize=(8, 5))
    times_to_compare = [0.0, 0.02, 0.05, 0.1]

    for t_target in times_to_compare:
        idx = np.argmin(np.abs(np.arange(n_steps_test + 1) * dt - t_target))
        plt.plot(
            x,
            u_ftcs[idx, :],
            linestyle="--",
            label=f"FTCS t={t_target:.2f}s",
        )
        plt.plot(
            x,
            u_cn[idx, :],
            linestyle="-",
            label=f"CN t={t_target:.2f}s",
        )

    plt.xlabel("x (m)")
    plt.ylabel("Temperature (K)")
    plt.title("FTCS vs Crank–Nicolson (bump diffusion test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/ftcs_vs_cn_bump.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
