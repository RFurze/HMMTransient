
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
    h_floor=1e-8, eps_solid=1e-8
):

    Lx, Ly = 1.0, 1.0
    N = 500
    mesh = GenerateSquareMesh_Scaled(N, Lx, Ly)

    xcoord, ycoord = x, y

    k=2
    h = A * (cos(2 * k * pi * xcoord) + cos(2 * k * pi * ycoord)) + B - (B-A)*x/1.5
    h = IfPos(h - h_floor, h, h_floor)  # safety floor

    # Field for contact region
    phi = h - hmin
    chi_fluid = IfPos(phi, 1.0, 0.0)  # 1 if h>hmin else 0

    # Create mask based on chi (includes a transition region i thought we needed
    # but it seems tow work without it)
    mask_core  = eps_solid
    mask_fluid = 1.0
    mask = smooth_levelset_mask(phi, delta_h, mask_core, mask_fluid)


    # Setting
    rho = 1.0
    mu  = 1.5
    Ux, Uy = 0, -0.8e-3


    K_base = rho * h**3 / (12.0 * mu)
    K = K_base * mask

    V0 = H1(mesh, order=2, dirichlet="left|right")
    V  = Periodic(V0)
    p, v = V.TnT()

    #Define weak form
    a = BilinearForm(V)
    a += K * grad(p) * grad(v) * dx

    f = LinearForm(V)
    f += mask * rho * h * (Ux/2 * grad(v)[0] + Uy/2 * grad(v)[1]) * dx

    a.Assemble()
    f.Assemble()

    gfp = GridFunction(V)
    gfp.Set(0.0)
    gfp.Set(0.0, definedon=mesh.Boundaries("left"))
    gfp.Set(500.0, definedon=mesh.Boundaries("right"))

    freedofs = V.FreeDofs()
    inv = a.mat.Inverse(freedofs, inverse="sparsecholesky")
    res = f.vec - a.mat * gfp.vec
    gfp.vec.data += inv * res

    # flux
    flux_cf = -K * grad(gfp)
    p_vis = chi_fluid * gfp

    vmin, vmax = masked_minmax(p_vis, chi_fluid, mesh, Lx, Ly, nsamp=200)
    print("Fluid-only pressure range:", vmin, vmax)

    plot_cf_matplotlib(p_vis, chi_fluid, mesh, Lx, Ly, nsamp=300, title="Pressure (fluid only)")
    plot_cf_matplotlib(mask, chi_fluid*0 + 1, mesh, Lx, Ly, nsamp=300, title="Mask")
    plot_cf_matplotlib(h, chi_fluid*0 + 1, mesh, Lx, Ly, nsamp=300, title="Film thickness h")

    print(f"  A={A}, B={B}, hmin={hmin}, delta_h={delta_h}, eps_solid={eps_solid}")
    return mesh, gfp, flux_cf, h, mask, chi_fluid


if __name__ == "__main__":
    mesh, p, flux, h, mask, chi = immersed_reynolds_solver_cosine_mask(
        A=0.002, B=5e-3, hmin=1.5e-3, delta_h=0, eps_solid=1e-8
    )
