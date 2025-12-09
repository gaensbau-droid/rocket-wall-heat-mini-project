"""
Config holding default parameters, domain settings,
and pulse schedule options.
"""

# Physical constants
kappa = 65.0            # thermal conductivity [W/(m*K)]
alpha = 1.7e-5          # thermal diffusivity [m^2/s]
hg = 1.5e4              # convective heat transfer coefficient [W/(m^2*K)]
Tg = 2500.0             # combustion gas stagnation temp [K]


# Domain settings
L = 0.01                # wall thickness [m]
nx = 100                # number of spatial cells
dx = L / (nx - 1)       # spatial step


# Time settings
dt = 1e-4               # time step [s]
nt = 15000              # number of time steps

# Pulse schedule
pulse_start = 0.0       # s
pulse_width  = 0.15     # s
pulse_period = 0.20     # s  # 0.15 on, 0.05 off

def pulse_is_on(t):
    """Return True if the heater/fire pulse is active at time t."""
    if t < pulse_start:
        return False
    phase = (t - pulse_start) % pulse_period
    return phase < pulse_width
