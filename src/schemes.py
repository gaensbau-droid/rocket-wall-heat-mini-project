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

def crank_nicolson(u0, nx, dx, dt, nt, alpha, bc_left, bc_right):
    """
    Crank–Nicolson solver for 1D heat diffusion with Dirichlet BCs.

    Parameters -
    
    u0 : ndarray
        Initial temperature distribution (size nx).
    nx : int
        Number of spatial grid points.
    dx : float
        Spatial step size.
    dt : float
        Time step size.
    nt : int
        Number of time steps.
    alpha : float
        Thermal diffusivity.
    bc_left : float
        Left boundary temperature (Dirichlet, constant in time).
    bc_right : float
        Right boundary temperature (Dirichlet, constant in time).

    Returns - 
    u : ndarray, shape (nt+1, nx)
        Temperature field at all time steps.
    """

    # allocate solution array
    u = np.zeros((nt + 1, nx))
    u[0, :] = u0.copy()

    r = alpha * dt / dx**2

    # interior points indices 1 .. nx-2  (size m)
    m = nx - 2
    if m <= 0:
        raise ValueError("Need at least 3 grid points for CN.")

    # Tridiagonal coefficients for A u^{n+1}_int = d
    # A has main diag (1 + r), off-diag -r/2
    a = -0.5 * r * np.ones(m - 1)      # sub-diagonal
    b = (1.0 + r) * np.ones(m)         # main diagonal
    c = -0.5 * r * np.ones(m - 1)      # super-diagonal

    # helper: Thomas algorithm for tridiagonal system
    def solve_tridiag(a, b, c, d):
        n = len(b)
        # forward sweep
        cp = np.zeros(n - 1)
        dp = np.zeros(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n - 1):
            denom = b[i] - a[i - 1] * cp[i - 1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
        dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-2])

        # back sub
        x = np.zeros(n)
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    # time stepping
    for n in range(nt):
        un = u[n, :].copy()

        # enforce Dirichlet BCs at current time
        un[0] = bc_left
        un[-1] = bc_right

        # build RHS for interior nodes (size m)
        d = np.zeros(m)
        for i in range(1, nx - 1):
            d_i = (1.0 - r) * un[i] + 0.5 * r * (un[i + 1] + un[i - 1])
            d[i - 1] = d_i

        # add BC contributions to RHS (first and last interior nodes)
        d[0]  += 0.5 * r * bc_left
        d[-1] += 0.5 * r * bc_right

        # solve for interior at time n+1
        u_int_next = solve_tridiag(a, b, c, d)

        # write back
        u[n + 1, 0]      = bc_left
        u[n + 1, 1:-1]   = u_int_next
        u[n + 1, -1]     = bc_right

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

