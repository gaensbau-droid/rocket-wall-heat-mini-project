# Rocket Wall Heat Mini-Project

Project: 1D transient heat conduction in a rocket thrust chamber wall using an explicit FTCS scheme with a Crank-Nicolson comparison test.

This code simulates a 1 cm thick metal wall when exposed to pulsed hot gas on the inner surface. 
It produces:
- wall temperature profiles at several times during operation
- a contour style plot of temperature vs. space and time
- an internal 1D diffusion test comparing FTCS and CN on a simple bump initial condition with error metrics

#1. Project Structure
Rocket-wall-heat-mini-project/
   src/
      _init__.py
      bc.py
      config.py
      diagnostics.py
      grid.py
      main.py
      problems.py     # (for future extensions)
      schemes.py
      simulate.py     # (for future CLI style interface)
   tests/
      test_convergence.py     # simple internal ftcs/cn check
   figures/
      wall_profiles_ftcs.png
      wall_contour_ftcs.png
      ftcs_cn_bump_diffusion.png
   report/
      .gitignore
      README.md
