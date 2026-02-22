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


# def q_re(xi, s, dt):
#     s += 1e-16  # avoid division by zero in cavitated regions
#     H, P, lmb, gradpx, gradpy, f_cav, Hdot, Pdot = xi
#     Hdt = H + Hdot * dt
#     Pdt = P + Pdot * dt
#     eta = 1e-3
#     rho = 1e3
#     qx = rho * s * (lmb * Hdt - (Hdt**3) * gradpx / (12 * eta * s))
#     qy = rho * s * (-(Hdt**3) * gradpy / (12 * eta * s))
#     qadv = rho * s * lmb * Hdt
#     qvisc = -rho * s * (Hdt**3) * gradpx
#     print(f"advective term {rho*s*lmb*Hdt})")
#     print(f"viscous term {-rho*s*(Hdt**3)*gradpx/(12*eta*s)}")
#     Pst = P
#     q = [qx, qy]
#     return q, qadv, qvisc

def q_re(xi, s, dt=0.0):
    s += 1e-16  # avoid division by zero in cavitated regions
    if len(xi) == 6:
        H, P, lmb, gradpx, gradpy, _ = xi
        Hdt = H
        Pdt = H
    else:
        H, P, lmb, gradpx, gradpy, _, Hdot, Pdot = xi
        Hdt = H + Hdot * dt
        Pdt = P + Pdot * dt
    eta = 1e-3
    qx = lmb * Hdt - (Hdt**3) * gradpx / (12 * eta * s)
    qy = -(Hdt**3) * gradpy / (12 * eta * s)
    Pst = P
    q = [qx, qy]
    return q, 0, 0

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
xmax = 75e-6  # domain width  (m)
ymax = 75e-6  # domain height (m)
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
def ht(x, y, xmax, ymax, H0, HT, T, Ux, Uy):
    Ah = 4e-9
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

pp0 = 3*400 / (2*np.pi * (14e-3)**2)
lmb0 = 12 * (1e-3) * (14e-3) / (pp0 * (5e-6)**2)
# ──────────────────────────────────────────────────────────────────────────────
# 3. Physical parameters  (dimensional unless noted)
# ──────────────────────────────────────────────────────────────────────────────
phys = micro_PhysicalParameters(
    Ux=-0.01918,
    Uy=-2.01231644387218E-18,  # plate half-velocities   (m s⁻¹)
    eta0=0.001,  # ambient viscosity       (Pa s)
    rho0=1000,  # ambient density         (kg m⁻³)
    alpha=1000.0,  # initial cavitation α-continuation value
    beta=26683.87432619197,  # Soderfjäll β parameter  (Pa) – will be rescaled
    beta_fraction=0.15,  # not used by the transient solver, but kept
    xmax=xmax,
    ymax=ymax,
    p0=-9602.26965351522,  # physical mean pressure  (Pa)
    dpdx=-6197527.73609277,
    dpdy=-336011.023171514,  # pressure gradients (Pa m⁻¹)
    H0=4.57347636818796E-06,  # nominal film thickness  (m)
    HT=-2.05772437248348E-09,  # film-thickness time derivative (m s⁻¹)
    PT=86005.3093889664,
    Tend=0.1,
)
print(f'Velocity = {phys.Ux:.6e} m s⁻¹')
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

Tf = 0.1  # final physical time (s)
n_steps = 10  # → Δt = Tf / n_steps

# print(f"Running {n_steps} steps up to T = {Tf:g} s …")
solver.run(Tf, n_steps)
# solver.run_steady()

# solver.run_steady()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Post-process homogenised outputs from the *final* time step
# ──────────────────────────────────────────────────────────────────────────────
(
    Qx,
    Qy,
    Pst,
    sst,
    taust_x,
    taust_y,
    p_max,
    p_min,
    max_h,
    min_h,
    qymax,
    qymin,
    fmax,
    fmin,
) = solver.spatial_homogenisation()
# print(f"sim Qx_adv = {solver.Qx_adv}, Qx_conv = {solver.Qx_conv}")
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
fcav = 0.686106617981718
xi = [phys.H0, phys.p0, phys.Ux / 2, phys.dpdx, phys.dpdy, fcav, phys.HT, phys.PT]
Fin = (sst + 1e-2) / (1 + 1e-2)
# print(f"Fin = {Fin:.6e} (dimensionless)")
Q_RE, Qadv, Qvisc = q_re(xi, Fin, Tf)

dQx = Qx - Q_RE[0]
dQy = Qy - Q_RE[1]
dPst = Pst - phys.p0
F_re = soderfjall_f(phys.p0, phys.beta)
dF = sst - F_re
# dQxadv = solver.Qx_adv - Qadv
# dQxvisc = solver.Qx_conv - Qvisc
# print(f'dQx = {dQx:.6e}, Qx_RE = {Q_RE[0]:.6e}, Qx = {Qx:.6e}, percentage change = {dQx / Q_RE[0] * 100:.6f}%')
# print(f'dQxadv = {dQxadv:.6e}, Qxre_adv = {Qadv:.6e}, Qxadv = {solver.Qx_adv:.6e}, percentage change = {dQxadv / Qadv * 100:.6f}%')
# print(f'dQxvisc = {dQxvisc:.6e}, Qxre_visc = {Qvisc:.6e}, Qxvisc = {solver.Qx_conv:.6e}, percentage change = {dQxvisc / (Qvisc+1e-16) * 100:.6f}%')
# print(f'Check total = {dQx - dQxadv - dQxvisc:.6e} m² s⁻¹')
print(f" Qre = {Q_RE}")
print(f"dQx = {dQx:.6e} m² s⁻¹")
print(f'Percentage change in dQx = {dQx / Q_RE[0] * 100:.6f}%')
print(f"dQy = {dQy:.6e} m² s⁻¹")

print(f"Pst = {Pst:.6e} Pa")
print(f"P_RE = {phys.p0:.6e} Pa")
print(f'dP = {Pst - phys.p0:.6e} Pa, percentage change = {(Pst - phys.p0) / phys.p0 * 100:.6f}%')
# print(f"sst = {sst:.6e} (dimensionless)")
# print(f"F_re = {F_re:.6e} (dimensionless)")
# print(f"dF = {dF:.6e} (dimensionless)")
