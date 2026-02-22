"""
demo_transient_parallel_plate.py
--------------------------------
Runs a 2 × 10-µm parallel-plate micro-simulation for 20 µs, using the
MicroTransientSolver_nondim class refactored from your original Code 1.

Place this file in the same folder as `micro_transient_solver.py`
(or make sure that file is on Python’s import path).
"""

# --- Standard & NGSolve -------------------------------------------------------
import os
from ngsolve import *
import numpy as np

# --- Your own helper modules --------------------------------------------------
from microscale.src.functions.micro_solver_transient import (
    MicroTransientSolver_nondim,
    micro_PhysicalParameters,
    micro_SolverSettings,
)

# If you already have this utility, import it; otherwise you can replace it with
# any square mesh generator of your choice.
from microscale.src.functions.micro_meshing import GenerateSquareMesh_Scaled


def q_re(xi, s, dt):
    s += 1e-16  # avoid division by zero in cavitated regions
    H, P, lmb, gradpx, gradpy, f_cav, Hdot, Pdot = xi
    Hdt = H + Hdot * dt
    Pdt = P + Pdot * dt
    eta = 1e-3
    rho = 1e3
    qx = rho * s * (lmb * Hdt - (Hdt**3) * gradpx / (12 * eta * s))
    qy = rho * s * (-(Hdt**3) * gradpy / (12 * eta * s))
    print(f"advective term {rho*s*lmb*Hdt})")
    print(f"viscous term {-rho*s*(Hdt**3)*gradpx/(12*eta*s)}")
    Pst = P
    q = [qx, qy]
    return q


def soderfjall_f(P, beta):
    if P > 0:
        return 1
    elif P < -beta:
        return 0
    else:
        return 1 - 2 * (P / beta) ** 3 - 3 * (P / beta) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# 1. Mesh
# ──────────────────────────────────────────────────────────────────────────────
xmax = 10e-6  # domain width  (m)
ymax = 10e-6  # domain height (m)
Ncells = 50  # number of elements per edge
mesh = GenerateSquareMesh_Scaled(Ncells, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Film-thickness function  h(x,y,t)
#    Must have signature  h(x, y, xmax, ymax, H0, HT, T)
#    Defined in dimensional coordinates
# ──────────────────────────────────────────────────────────────────────────────
# def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
#     """
#     Constant film for this demo (no squeeze-film term, no waviness).
#     Replace with any spatial/temporal formula you like.
#     """
#     A = 1e-8
#     x_dim, y_dim = x * xmax, y * xmax
#     return H0 + HT * T + 0.5*A*sin(2 * np.pi * (x_dim + Ux*T) / xmax) * sin(2 * np.pi * (y_dim + Uy*T) / ymax)

var = 0.01
mult_list = np.linspace(0.1, 0.95, 10)
dQx_list = []
dPst_list = []
for var in mult_list:
    # ──────────────────────────────────────────────────────────────────────────────
    # 3. Physical parameters  (dimensional unless noted)
    # ──────────────────────────────────────────────────────────────────────────────
    phys = micro_PhysicalParameters(
        Ux=-1.917999999999999566e-02,
        Uy=0,  # plate half-velocities   (m s⁻¹)
        eta0=0.001,  # ambient viscosity       (Pa s)
        rho0=1000,  # ambient density         (kg m⁻³)
        alpha=1000.0,  # initial cavitation α-continuation value
        beta=8347.612189010071,  # Soderfjäll β parameter  (Pa) – will be rescaled
        beta_fraction=0.15,  # not used by the transient solver, but kept
        xmax=xmax,
        ymax=ymax,
        p0=3295915,  # physical mean pressure  (Pa)
        dpdx=0,
        dpdy=0,  # pressure gradients (Pa m⁻¹)
        H0=5.800000000000000409e-08,  # nominal film thickness  (m)
        HT=0,  # film-thickness time derivative (m s⁻¹)
        PT=0,
        Tend=1e-1,
    )

    def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
        Ah = var * phys.H0
        kx = 1
        ky = 1
        x_d, y_d = x * xmax, y * ymax
        hmin0 = H0 - Ah
        hmin1 = H0 - Ah + HT * T
        if hmin0 < 0 or hmin1 < 0:
            print(
                f"Warning: Negative film thickness at T={T}, hmin0 = {hmin0}, hmin1 = {hmin1}."
            )
            sys.stdout.flush()
        out = (
            H0
            + HT * T
            + (Ah * sin(kx * 2 * pi * (x_d + Ux * T) / xmax) * sin(ky * 2 * pi * y_d / ymax))
        )
        return out



    # ──────────────────────────────────────────────────────────────────────────────
    # 4. Non-linear/Newton solver controls
    # ──────────────────────────────────────────────────────────────────────────────
    settings = micro_SolverSettings(
        newton_damping=1.0,
        max_iterations=200,
        error_tolerance=1e-10,
        alpha_reduction_factor=0.1,
        alpha_threshold=1e-2,
    )

    # ──────────────────────────────────────────────────────────────────────────────
    # 5. Instantiate & run
    # ──────────────────────────────────────────────────────────────────────────────
    solver = MicroTransientSolver_nondim(
        mesh=mesh,
        physical_params=phys,
        solver_settings=settings,
        k=2,  # polynomial order of H¹ space
        ht=ht,  # film-thickness callback
        target_dof=0,  # corner node to pin pressure to zero (unique solution)
        export_vtk=True,
        output_dir=os.path.join(os.getcwd(), "output"),  # output directory
    )

    Tf = 1e-8  # final physical time (s)
    n_steps = 2  # → Δt = Tf / n_steps

    # print(f"Running {n_steps} steps up to T = {Tf:g} s …")
    solver.run(Tf, n_steps)
    # solver.run_steady()

    # solver.run_steady()

    # ──────────────────────────────────────────────────────────────────────────────
    # 6. Post-process homogenised outputs from the *final* time step
    # ──────────────────────────────────────────────────────────────────────────────
    Qx, Qy, Pst, sst, taustx, tausty = solver.spatial_homogenisation()
    print(f"sim Qx_adv = {solver.Qx_adv}, Qx_conv = {solver.Qx_conv}")
    print(
        """=== Spatially-averaged results (final step) ===
    ⟨Qx⟩ = {0:.6e}  m² s⁻¹
    ⟨Qy⟩ = {1:.6e}  m² s⁻¹
    ⟨P ⟩ = {2:.6e}  Pa
    ⟨s ⟩ = {3:.6e}  (dimensionless)
    """.format(
            Qx, Qy, Pst, sst
        )
    )
    fcav = 1
    xi = [phys.H0, phys.p0, phys.Ux / 2, phys.dpdx, phys.dpdy, fcav, phys.HT, phys.PT]
    Fin = (sst + 1e-2) / (1 + 1e-2)
    print(f"Fin = {Fin:.6e} (dimensionless)")
    Q_RE = q_re(xi, Fin, Tf)

    dQx = (Qx - Q_RE[0])/Q_RE[0]
    dQy = Qy - Q_RE[1]
    dPst = (Pst - phys.p0)/phys.p0
    F_re = soderfjall_f(phys.p0, phys.beta)
    dF = sst - F_re
    dQx_list.append(dQx)
    dPst_list.append(dPst)


import matplotlib.pyplot as plt
#Create two subplots
fig, ax1 = plt.subplots()
# Plot dQx vs var on the first y-axis
ax1.plot(mult_list, dQx_list, 'b-', label='dQx')
ax1.set_xlabel('Ah/H0')
ax1.set_ylabel('change in dQx', color='b')
ax1.tick_params(axis='y', labelcolor='b')
# Create a second y-axis for dPst
ax2 = ax1.twinx()
ax2.plot(mult_list, dPst_list, 'r-', label='dPst')
ax2.set_ylabel('change in dPst', color='r')
ax2.tick_params(axis='y', labelcolor='r')
# Add a title and show the plot
plt.title('Transient Solver Results')
plt.show()
plt.savefig('transient_results_varAh.png')