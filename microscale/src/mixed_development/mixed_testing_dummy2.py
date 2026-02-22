"""
Added pressure based deformation, fixed film thickness
No near-periodic boundary conditions
No point constraint
No cavitation
"""



def GenerateSquareMesh_Scaled(N, Lx, Ly):

    N = int(N // 2)
    quads = True

    mesh = NGMesh()
    mesh.SetGeometry(unit_square)
    mesh.dim = 2
    pnums = []
    coordinates = []
    slave_points_y = {}
    matched_points = set()

    # points
    for i in range(N + 1):
        for j in range(N + 1):
            x = (i / N) * Lx
            y = (j / N) * Ly
            point = Pnt(x, y, 0.0)
            pnums.append(mesh.Add(MeshPoint(point)))
            coordinates.append(point)

            if j == 0:
                slave_points_y[i] = pnums[-1]

    mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    mesh.SetMaterial(1, "mat")

    # elements
    for j in range(N):
        for i in range(N):
            if quads:
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + j * (N + 1)],
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
            else:
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + j * (N + 1)],
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )
                mesh.Add(
                    Element2D(
                        1,
                        [
                            pnums[i + (j + 1) * (N + 1)],
                            pnums[i + 1 + (j + 1) * (N + 1)],
                            pnums[i + 1 + j * (N + 1)],
                        ],
                    )
                )

    # horizontal boundaries
    for i in range(N):
        # top: j = N
        mesh.Add(
            Element1D(
                [pnums[i + N * (N + 1)], pnums[i + 1 + N * (N + 1)]],
                index=1,
            )
        )
        # bottom: j = 0
        mesh.Add(
            Element1D(
                [pnums[i], pnums[i + 1]],
                index=2,
            )
        )

    # vertical boundaries
    for j in range(N):
        # left: i = 0
        mesh.Add(
            Element1D(
                [pnums[j * (N + 1)], pnums[(j + 1) * (N + 1)]],
                index=3,
            )
        )
        # right: i = N
        mesh.Add(
            Element1D(
                [pnums[j * (N + 1) + N], pnums[(j + 1) * (N + 1) + N]],
                index=4,
            )
        )

    # Identify points for the y-direction (top identified to bottom)
    for i in range(N + 1):
        for j in range(N + 1):
            if j == N:
                slave = slave_points_y[i]
                mesh.AddPointIdentification(
                    pnums[i * (N + 1) + j], slave, identnr=2, type=2
                )
                matched_points.add(pnums[i * (N + 1) + j])
                matched_points.add(slave)

    # Boundary names
    mesh.SetBCName(2, "top")
    mesh.SetBCName(3, "bottom")
    mesh.SetBCName(1, "left")
    mesh.SetBCName(0, "right")

    meshout = ngsolve.Mesh(mesh)
    return meshout

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import math

def masked_minmax(cf, mask_cf, mesh, Lx, Ly, nsamp=200):
    import numpy as np

    vals = []

    for i in range(nsamp+1):
        xk = Lx * i / nsamp
        for j in range(nsamp+1):
            yk = Ly * j / nsamp

            mip = mesh(xk, yk)
            if float(mask_cf(mip)) > 0.5:   # “fluid” region only
                vals.append(float(cf(mip)))

    if not vals:
        raise RuntimeError("No points found in masked region (check mask_cf/Lx/Ly)")
    vals = np.array(vals)
    return float(vals.min()), float(vals.max())

from netgen.meshing import Mesh as NGMesh, MeshPoint, Element2D, Element1D, Pnt, FaceDescriptor
from netgen.csg import *
import ngsolve
from ngsolve.solvers import *
from ngsolve import *
from ngsolve.webgui import Draw
import math
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def plot_cf_matplotlib(cf, mask_cf, mesh, Lx, Ly, nsamp=300, title=""):
    xs = np.linspace(0, Lx, nsamp+1)
    ys = np.linspace(0, Ly, nsamp+1)

    Z = np.full((nsamp+1, nsamp+1), np.nan, dtype=float)

    for i, xk in enumerate(xs):
        for j, yk in enumerate(ys):
            mip = mesh(xk, yk)
            if float(mask_cf(mip)) > 0.5:
                Z[j, i] = float(cf(mip))

    plt.figure()
    # contourf handles NaNs fine (they appear blank)
    plt.contourf(xs, ys, Z, levels=50)
    plt.colorbar()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(title)
    plt.show()
    plt.savefig(title+".png", dpi=200, bbox_inches="tight")

def smooth_levelset_mask(phi, delta, mask_core, mask_fluid):
    t = (phi + delta) / (2 * delta)
    mask_transition = mask_core + (mask_fluid - mask_core) * t

    mask = IfPos(phi - delta,
                 mask_fluid,
                 IfPos((-delta) - phi,
                       mask_core,
                       mask_transition))
    return mask


def immersed_reynolds_solver_cosine_mask(
    A=0.002, B=1e-3, hmin=7.5e-4, delta_h=5e-5,
    h_floor=1e-8, eps_solid=1e-8,
    # NEW:
    kappa_p=0.0,                 # coupling strength in h = base + kappa_p * p
    max_outer=50,                # max pressure↔h iterations
    tol_outer=1e-8,              # convergence tol on pressure change (relative)
    omega=1.0,                   # under-relaxation (1.0 = none)
    p_init_scalar=500.0          # initial "pressure" used in h for the first outer iter
):

    Lx, Ly = 1.0, 1.0
    N = 500
    mesh = GenerateSquareMesh_Scaled(N, Lx, Ly)

    xcoord, ycoord = x, y

    # cosine wavemode (kept as your original)
    k_mode = 2

    # Setting
    rho = 1.0
    mu  = 1.5
    Ux, Uy = 0, -0.8e-3

    # Space (kept as your original)
    V0 = H1(mesh, order=2, dirichlet="left|right")
    V  = Periodic(V0)

    # Pressure unknown / storage
    p_gf = GridFunction(V)   # current iterate (p_old)
    p_gf.Set(0.0)

    # Helper: build everything that depends on "pressure used in h"
    def build_h_mask_K(p_for_h):
        # base thickness (your original profile)
        h_base = A * (cos(2 * k_mode * pi * xcoord) + cos(2 * k_mode * pi * ycoord)) + B #- (B - A) * xcoord / 1.5

        # NEW coupling term: + kappa_p * pressure
        h_cf = h_base + kappa_p * p_for_h

        # safety floor
        h_cf = IfPos(h_cf - h_floor, h_cf, h_floor)

        # contact / fluid indicator
        phi = h_cf - hmin
        chi_fluid = IfPos(phi, 1.0, 0.0)

        # smooth mask (same as before; delta_h=0 is allowed but then transition is sharp-ish)
        mask_core  = eps_solid
        mask_fluid = 1.0
        mask_cf = smooth_levelset_mask(phi, delta_h, mask_core, mask_fluid)

        # mobility
        K_base = rho * h_cf**3 / (12.0 * mu)
        K_cf = K_base * mask_cf

        return h_cf, mask_cf, chi_fluid, K_cf

    # Outer fixed-point loop
    last_rel = None
    for it in range(max_outer):

        # 1) choose what "pressure" goes into h this iteration
        if it == 0:
            # per your request: initially use the boundary constraint value (500) everywhere in h
            p_for_h = p_init_scalar
        else:
            # then use the current pressure field
            p_for_h = p_gf

        # 2) rebuild h, mask, K
        h, mask, chi_fluid, K = build_h_mask_K(p_for_h)

        # 3) rebuild & assemble weak forms (depend on h and K)
        p, v = V.TnT()

        a = BilinearForm(V)
        a += K * grad(p) * grad(v) * dx

        f = LinearForm(V)
        f += mask * rho * h * (Ux/2 * grad(v)[0] + Uy/2 * grad(v)[1]) * dx

        a.Assemble()
        f.Assemble()

        # 4) solve linear system for p_new
        p_new = GridFunction(V)
        p_new.Set(0.0)
        p_new.Set(0.0,   definedon=mesh.Boundaries("left"))
        p_new.Set(1000.0, definedon=mesh.Boundaries("right"))

        freedofs = V.FreeDofs()
        inv = a.mat.Inverse(freedofs, inverse="sparsecholesky")
        res = f.vec - a.mat * p_new.vec
        p_new.vec.data += inv * res

        # 5) convergence check on pressure change (after "deformation" update)
        #    We'll use an L2-like vector norm on the DOF vector.
        diff_vec = p_new.vec.CreateVector()
        diff_vec.data = p_new.vec - p_gf.vec

        num = diff_vec.Norm()
        den = max(p_new.vec.Norm(), 1e-30)
        rel = num / den
        last_rel = rel

        print(f"[OUTER] it={it:02d}  rel_dp={rel:.3e}")

        # 6) accept update (with optional under-relaxation)
        if omega >= 1.0:
            p_gf.vec.data = p_new.vec
        else:
            p_gf.vec.data = (1.0 - omega) * p_gf.vec + omega * p_new.vec

        if it > 0 and rel < tol_outer:
            print(f"[OUTER] converged at it={it:02d} with rel_dp={rel:.3e}")
            break

    # final outputs (same spirit as before)
    flux_cf = -K * grad(p_gf)
    p_vis = chi_fluid * p_gf

    vmin, vmax = masked_minmax(p_vis, chi_fluid, mesh, Lx, Ly, nsamp=200)
    print("Fluid-only pressure range:", vmin, vmax)
    print(f"Final outer rel_dp: {last_rel}")

    plot_cf_matplotlib(p_vis, chi_fluid, mesh, Lx, Ly, nsamp=300, title="Pressure (fluid only)")
    plot_cf_matplotlib(mask, chi_fluid*0 + 1, mesh, Lx, Ly, nsamp=300, title="Mask")
    plot_cf_matplotlib(h, chi_fluid*0 + 1, mesh, Lx, Ly, nsamp=300, title="Film thickness h (coupled)")

    print(f"  A={A}, B={B}, hmin={hmin}, delta_h={delta_h}, eps_solid={eps_solid}")
    print(f"  kappa_p={kappa_p}, max_outer={max_outer}, tol_outer={tol_outer}, omega={omega}")
    return mesh, p_gf, flux_cf, h, mask, chi_fluid



if __name__ == "__main__":
    mesh, p, flux, h, mask, chi = immersed_reynolds_solver_cosine_mask(
        A=0.002, B=5e-3, hmin=1.5e-3, delta_h=0, eps_solid=1e-8,
        kappa_p=5e-6,      # <-- choose a sensible magnitude for your problem
        max_outer=30,
        tol_outer=1e-8,
        omega=0.2          # <-- under-relaxation often helps if coupling is strong
    )