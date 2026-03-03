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
    OUTPUT_DIR="BaseCase020326",
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
    k_spring=1e20,
)

ideal_film_thickness = FilmThicknessParams(
    Ah=2.25e-7,  # 1.25e-7,
    kx=1,
    ky=1,
)


def angular_velocity(t: float) -> float:
    """Default angular velocity evolution.

    Users may modify this function to prescribe a different time-dependent
    angular velocity profile.

    IMPORTANT — velocity convention
    --------------------------------
    This function must return the **mean** angular velocity of the contact,
    i.e. half the surface angular velocity of the rotating body:

        ω_mean = ω_surface / 2

    For a ball-in-cup contact where only the ball rotates at ω_surface and
    the cup is fixed, supply ω_surface / 2 here.  The macroscale Reynolds
    equation uses this value directly as the Couette velocity (U_mean = Rc·ω),
    and the microscale solver doubles the received linear velocity to recover
    the true surface velocity (U_surface = 2·U_mean) for shear-stress
    calculations.  Passing ω_surface instead of ω_mean would double both the
    hydrodynamic pressure and the predicted friction force.

    The profile should be input with dimensions rad/s (mean).
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
    Rnewton_relaxation_parameter=0.5,
    R_krylov_rtol=1e-5,
    load_balance_rtol=1e-5,
    xi=1e6,
    bc_tol=1e-4,
    Id=1,
    Tend=4.0,
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
    k_spring=1e20,
)

micro_solver = MicroSolverSettings(
    dt=runtime.DT, #Note that this is just stupid naming - dt in micro is actually the end time
    tsteps=10,
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
    coupling_tol=1e-5,
    load_balance_tol=1e-5,
    scaling_factor=1e-3,
)

# ------------------------------------------------------------------
# MLS parameters
# ------------------------------------------------------------------
# MLS_THETA = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
# MLS_THETA = np.array([10000, 10000, 10000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
MLS_THETA = np.array([3000, 3000, 3000, 2000, 2000, 2000, 2000, 2000, 2000, 2000])  # softer kernel for better 6D coverage
MLS_DEGREE = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


# Coupling defaults
ND_FACTOR = 0.75
RO_THETA = 20

# ------------------------------------------------------------------
# EDAS (Error-Driven Adaptive Sampling) parameters — transient only
# ------------------------------------------------------------------
from config.dataclasses import EDASSettings

edas = EDASSettings(
    batch_size=200,          # micro sims per refinement pass (was 500; smaller for better error-guided placement)
    max_budget=6000,         # total micro sim budget per coupling iteration
    max_refine_passes=10,    # refinement sub-iterations (was 4; more passes compensate for smaller batches)
    error_target=0.05,       # stop refining when max pointwise error < this
    alpha_blend=0.5,         # blend weight: coverage (1) vs prediction error (0)
    delta_min_quantile=0.02, # quantile of 1-NN distance for minimum spacing (was 0.1; q=0.1 wasted 47% of budget)
    lambda_decay=2.0,        # time-decay rate for relevance weighting
    sigma_spatial=0.3,       # spatial relevance lengthscale (normalised space)
    relevance_prune_threshold=0.01,  # prune training points below this weight
    r0_quantile=0.15,        # 1-NN quantile for cold-start coverage radius (was 0.25; denser initial sampling)
    coupling_decay=0.5,      # per-coupling-iteration relevance decay (0.5 = halve weight each iteration)
    edas_coupling_threshold=0.01,  # only run EDAS refinement when coupling_error < this
)
