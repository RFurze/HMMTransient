import os
import numpy as np
import matplotlib.pyplot as plt
from microscale.src.functions.micro_solver_beta import *
from microscale.src.functions.micro_meshing import GenerateSquareMesh_Scaled
import time


def q_re(xi):
    H, P, lmb, gradpx, gradpy = xi
    eta = 1e-3
    rho = 1e3
    # Note that the components of lmb are actually Um, Vm, so 0.5 * Ux and 0.5 * Vy
    qx = rho * (lmb * H - (H**3) * gradpx / (12 * eta))
    qy = rho * (-(H**3) * gradpy / (12 * eta))
    qz = np.zeros_like(qx)
    Pst = P
    q = np.column_stack((qx, qy, qz))
    return q


def soderfjall_f(P, beta):
    if P > 0:
        print("P > 0")
        return 1
    elif P < -beta:
        print(f"P {P} < -beta {-beta}")
        return 0
    else:
        return 1 - 2 * (P / beta) ** 3 - 3 * (P / beta) ** 2


def run_micro(H, P, lmb, gradp, f, mesh):
    start = time.time()
    # flush_print(f'run_micro called on rank={rank}, id={id}, perm={perm}')
    start_time = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    xi = [H, P, lmb[0], gradp[0], gradp[1], f]
    betain = 8347.603341942322
    physical_params = micro_PhysicalParameters(
        Ux=lmb[0],
        Uy=lmb[1],
        eta0=1e-3,
        rho0=1000,
        alpha=1000.0,
        beta=betain,
        beta_fraction=0.05,
        xmax=10e-6,
        ymax=10e-6,
        p0=P,
        dpdx=gradp[0],
        dpdy=gradp[1],
        H0=H,
    )

    solver_settings = micro_SolverSettings(
        newton_damping=1.0,
        max_iterations=200,
        error_tolerance=1e-6,
        alpha_reduction_factor=0.1,
        alpha_threshold=0.0001,
    )

    # Define an analytical film thickness for testing
    def ht(x, y, xmax, ymax, H0):
        Ah = 1e-8
        kx = 1
        ky = 1
        out = H0  # + (0.5*Ah*sin(kx*2*pi*x/xmax)*sin(ky*2*pi*y/ymax))
        return out

    def nondim_h(x, y, xmax, ymax, H0, ht=ht):
        return ht(x * xmax, y * ymax, xmax, ymax, H0) / H0

    # Initialize the solver
    logger.info("Initializing the solver...")
    solver = MicroSolver_nondim(
        mesh, physical_params, solver_settings, k=3, ht=nondim_h
    )
    f_re = soderfjall_f(P, betain)
    # Run the solver
    logger.info("Running the solver...")
    solver.run()
    # logger.info(f"Solver finished in {time.time() - start:.2f} seconds.")
    # Post-process results
    logger.info("Post-processing results...")
    solver.post_process()
    Qx, Qy, Pst, sst = solver.spatial_homogenisation()

    dF = sst - f_re

    Q = np.column_stack((Qx, Qy, 0))
    Q_re = q_re(xi, sst)
    delta_Q = Q - Q_re
    delta_P = Pst - P

    print(f"delta_Q: {delta_Q}, delta_P: {delta_P}")
    print(f"Q_re: {Q_re}")
    print(f"Q: {Q}")
    print(f"Pst: {Pst}")
    print(f"P : {P}")
    print(f"sst: {sst}")
    print(f"f_re: {f_re}")
    print(f"f_in : {xi[5]}")
    print(f"dF: {dF}")
    print(f"f_in - f_re: {xi[5] - f_re}")

    duration = time.time() - start

    return duration, Q, Pst


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
INPUT_DIR = os.path.join(BASE_DIR, "data/input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

mesh = GenerateSquareMesh_Scaled(100, 1, 1)

# Initialise the solver
H = 5e-7
P = 1e5
lmb = np.asarray([0.02, 0.0])
gradp = np.asarray([1e9, 0.0])
f = 0.125885815915104


def q_re(xi, s):
    H, P, lmb, gradpx, gradpy, sst = xi
    s += 1e-16
    eta = 1e-3
    rho = 1e3
    # Note that the components of lmb are actually Um, Vm, so 0.5 * Ux and 0.5 * Vy (no factor of 1/2 on the advection term)
    qx = rho * s * ((lmb * H) - (H**3) * gradpx / (12 * eta * s))
    qx_adv = rho * s * lmb * H
    qx_conv = -rho * s * (H**3) * gradpx / (12 * eta * s)
    print(f"qx_re_adv: {qx_adv}, qx_re_conv: {qx_conv}")
    qy = rho * s * ((0 * H) - (H**3) * gradpy / (12 * eta * s))
    qz = np.zeros_like(qx)
    Pst = P
    q = np.column_stack((qx, qy, qz))
    return q


xi = [H, P, lmb[0], gradp[0], gradp[1], f]
# q_re_xi = q_re(xi)
# print(f'q_re_xi: {q_re_xi}')

duration, q, Pst = run_micro(H, P, lmb, gradp, f, mesh)


print(f"Time taken: {duration}")
print(f"Q: {q}, Pst: {Pst}")
