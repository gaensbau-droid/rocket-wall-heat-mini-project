import numpy as np

def ftcs(u0, nx, dx, dt, nt, alpha, bc_left, bc_right):
    """
    FTCS solver for 1D heat diffusion.

    Parameters -
    
    u0 : ndarray
        Initial temperature distribution (size nx)
    nx : int
        Number of spatial grid points
    dx : float
        Spatial step size
    dt : float
        Time-step size
    nt : int
        Number of time steps
    alpha : float
        Thermal diffusivity
    bc_left : float
        Left boundary temperature (Dirichlet)
    bc_right : float
        Right boundary temperature (Dirichlet)

    Returns - 
    u : ndarray, shape (nt+1, nx)
        Temperature field at all time steps
    """

    # stability condition
    Fo = alpha * dt / dx**2
    if Fo > 0.5:
        raise ValueError(
            f"FTCS unstable: alpha*dt/dx^2 = {Fo:.4f} > 0.5. "
            "Reduce dt or increase dx."
        )

    # allocate solution array
    u = np.zeros((nt + 1, nx))
    u[0, :] = u0.copy()

    # time step
    for n in range(nt):
        un = u[n, :].copy()

        # boundary (simple Dirichlet for this test)
        u[n+1, 0] = bc_left
        u[n+1, -1] = bc_right

        # updates inside
        for i in range(1, nx - 1):
            u[n+1, i] = (
                un[i]
                + Fo * (un[i+1] - 2.0 * un[i] + un[i-1])
            )

    return u

def heat_wall_ftcs(T0, nx, dx, dt, nt, alpha, kappa, hg, Tg, pulse_is_on):
    """
    Physical FTCS solver for the rocket thrust chamber wall.

    - 1D heat equation in the wall
    - Convective BC at x=0 when the thruster is firing
    - Zero-flux (Neumann) BC at x=0 when off
    - Zero-flux (Neumann) BC at x=L always
    """

    # Allocate solution array: time × space
    T = np.zeros((nt + 1, nx))
    T[0, :] = T0.copy()

    r = alpha * dt / dx**2
    if r > 0.5:
        print(f"WARNING: FTCS stability parameter r={r:.4f} > 0.5")

    for n in range(nt):
        Tn = T[n, :].copy()
        t_n = n * dt

        # Left boundary (x=0)
        if pulse_is_on(t_n):
            # Convective BC: -k dT/dx = hg (Tg - T_wall)
            # Ghost node T_{-1} = T_1 + (2 dx hg / kappa) (Tg - T_0)
            TghostL = Tn[1] + (2.0 * dx / kappa) * hg * (Tg - Tn[0])
        else:
            # Thruster off: adiabatic → dT/dx = 0 → T_{-1} = T_1
            TghostL = Tn[1]


        T[n+1, 0] = Tn[0] + r * (Tn[1] - 2.0 * Tn[0] + TghostL)

        # Interior nodes
        for i in range(1, nx - 1):
            T[n+1, i] = Tn[i] + r * (Tn[i+1] - 2.0 * Tn[i] + Tn[i-1])

        # Right boundary (x=L): Neumann 0
        # dT/dx = 0 → T_{N} = T_{N-2}
        TghostR = Tn[-2]
        T[n+1, -1] = Tn[-1] + r * (TghostR - 2.0 * Tn[-1] + Tn[-2])

    return T

