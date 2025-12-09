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
