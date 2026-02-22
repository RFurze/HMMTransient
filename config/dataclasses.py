from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any


@dataclass
class MaterialParams:
    Rc: float
    c: float
    rho0: float
    eta0: float
    t0: float
    load_mag: float
    load_orientation: Sequence[float]
    eccentricity0: Sequence[float]
    E: float
    nu: float
    k_spring: float


@dataclass
class MeshParams:
    cupmeshdir: str
    ballmeshdir: str
    CupScale: Sequence[float]
    BallScale: Sequence[float]
    CupDisplace: Sequence[float]
    delta: float
    tol: float


@dataclass
class SolverParams:
    Rnewton_max_iterations: int
    Rnewton_rtol: float
    Rnewton_atol: float
    Rnewton_relaxation_parameter: float
    R_krylov_rtol: float
    load_balance_rtol: float
    xi: float
    bc_tol: float
    Id: Any
    Tend: float
    dt: float
    t_export_interval: float
    angular_velocity_fn: Any
    dynamic_load_fn: Any
    K_matrix_dir: str


@dataclass
class MicroPhysicalParams:
    eta0: float
    rho0: float
    alpha: float
    beta_fraction: float
    xmax: float
    ymax: float
    k_spring: float


@dataclass
class MicroSolverSettings:
    dt: float
    tsteps: int
    newton_damping: float
    max_iterations: int
    error_tolerance: float
    alpha_reduction: float
    alpha_threshold: float
    ncells: int
    progress_interval: int


@dataclass
class FilmThicknessParams:
    Ah: float
    kx: float
    ky: float


@dataclass
class TransientSettings:
    log_file: str
    output_dir: Path
    coupling_tol: float
    load_balance_tol: float
    scaling_factor: float


@dataclass
class RuntimeSettings:
    OUTPUT_DIR: str
    MAX_LB_ITERS: int
    MAX_COUPLING_ITERS: int
    TEND: float
    DT: float
    T0: float
