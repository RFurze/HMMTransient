#!/usr/bin/env python3
"""Central configuration for macroscale, microscale and runtime parameters."""

"""
LAST UPDATE CHANGED ALPHA LOGIC IN FINAL POLISH STEP
"""

from pathlib import Path
# from fenics import Identity
import numpy as np

from config.dataclasses import (
    MaterialParams,
    MeshParams,
    SolverParams,
    MicroPhysicalParams,
    FilmThicknessParams,
    MicroSolverSettings,
    TransientSettings,
    RuntimeSettings,
)

# ------------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------------
runtime = RuntimeSettings(
    OUTPUT_DIR="PenaltyDebug9",
    MAX_LB_ITERS=20,
    MAX_COUPLING_ITERS=20,
    TEND=4.00,
    DT=0.05,
    T0=0.0,
)


# ------------------------------------------------------------------
# Macroscale parameters
# ------------------------------------------------------------------
material = MaterialParams(
    Rc=25e-3,
    c=50e-6,
    rho0=1e3,
    eta0=1e-2,
    t0=1,
    load_mag=800, #200
    load_orientation=[0, 0, 1],
    eccentricity0=[0.0, 0.0, 0.9950523437], #0.9541235992
    E=105e9,
    nu=0.3,
    k_spring=5e13,
)

ideal_film_thickness = FilmThicknessParams(
    Ah=2.75e-7,  # 1.25e-7,
    kx=1,
    ky=1,
)


def angular_velocity(t: float) -> float:
    """Default angular velocity evolution.

    Users may modify this function to prescribe a different time-dependent
    angular velocity profile.

    The profile should be input with dimensions rad/s.
    """
    # return np.round(-5 * np.pi * np.pi * np.cos(t * 2 * np.pi) / 18, 2) # Friction test
    return 2  # round(5 * (0.4 * np.cos(t * 2 * np.pi) + 0.6), 4)


def dynamic_load(t: float) -> float:
    """Default dynamic load evolution.

    Users may modify this function to prescribe a different time-dependent
    dynamic load profile.

    The profile should be input with dimensions N.
    """
    # calculate the current position in a 1s cycle
    # ti = t % 1
    # if 0.22 <= ti <= 0.78:
    #     load = -21.68 * (ti - 0.5) ** 2 + 2
    # else:
    #     load = 0.3
    # load_out = load * 1/2 * 500
    # return load_out #Friction test
    # return material.load_mag * (0.4 * np.sin(t * 2 * np.pi) + 0.6)
    return material.load_mag * (0.45 * np.sin(t * 2 * np.pi) + 0.55)


mesh = MeshParams(
    cupmeshdir="meshing/data/HydroMesh7_Cut3d.xdmf",
    ballmeshdir="../../mesh/Ball4.xdmf",
    CupScale=[1, 1, 1],
    BallScale=[(material.Rc - material.c) / material.Rc] * 3,
    CupDisplace=[0.0, 0.0, 0.0],
    delta=0.008,
    tol=1e-4,
)

solver = SolverParams(
    Rnewton_max_iterations=1000,
    Rnewton_rtol=1e-6,
    Rnewton_atol=1e-8,
    Rnewton_relaxation_parameter=0.6,
    R_krylov_rtol=1e-5,
    load_balance_rtol=1e-5,
    xi=1e6,
    bc_tol=1e-4,
    Id=1,
    Tend=3.0,
    dt=0.05,
    t_export_interval=0.1,
    angular_velocity_fn=angular_velocity,
    dynamic_load_fn=dynamic_load,
    K_matrix_dir="deformation/M25Thin-Mesh7/HydroMesh7_Cut3d_infM_combined_full.npz",
)


# Steady-state solver tolerances
STEADY_COUPLING_TOL = 1e-5
STEADY_LOAD_BALANCE_TOL = 1e-5
STEADY_SCALING_FACTOR = 1e-2

# ------------------------------------------------------------------
# Microscale parameters
# ------------------------------------------------------------------
micro_physical = MicroPhysicalParams(
    eta0=1e-2,
    rho0=1e3,
    alpha=1000.0,
    beta_fraction=0.05,
    xmax=7.5e-5,
    ymax=7.5e-5,
    k_spring=5e13,
)

micro_solver = MicroSolverSettings(
    dt=runtime.DT, #Note that this is just stupid naming - dt in micro is actually the end time
    tsteps=8,
    newton_damping=1.0,
    max_iterations=200,
    error_tolerance=1e-6,
    alpha_reduction=0.1,
    alpha_threshold=1e-3,
    ncells=40,
    progress_interval=100,
)

# ------------------------------------------------------------------
# Transient workflow defaults
# ------------------------------------------------------------------
transient = TransientSettings(
    log_file="HMM_Transient.log",
    output_dir=Path("data/output/hmm_job"),
    coupling_tol=1e-4,
    load_balance_tol=1e-5,
    scaling_factor=1e-3,
)

# ------------------------------------------------------------------
# MLS parameters
# ------------------------------------------------------------------
# MLS_THETA = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
MLS_THETA = np.array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
MLS_DEGREE = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


# Coupling defaults
ND_FACTOR = 0.6
RO_THETA = 20
