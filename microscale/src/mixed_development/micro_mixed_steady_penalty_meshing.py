"""
Shifted periodic BCs with Reynolds
Penalty cavitation
Point constraint

CUT-MESH VERSION:
Instead of "masking" coefficients where H < hmin, we:
  1) mark cells as "fluid" if H(cell midpoint) > hmin
  2) create SubMesh consisting only of those fluid cells
  3) solve Reynolds on SubMesh (so the masked region becomes an internal boundary)

Notes:
- The internal boundary gets natural no-flux BC (since we use the standard weak form
  without adding boundary integrals).
- Shifted periodic BCs are applied on the outer boundaries (x=0/1, y=0/1) of the SubMesh.
- If the cut-out touches the outer boundary, periodic mapping can become ill-defined; this
  script assumes the "solid" region is interior (typical contact patch).
"""

from dolfin import *
import numpy as np

parameters["allow_extrapolation"] = True

# -----------------------------
# Mesh / params
# -----------------------------
n = 200
mesh = UnitSquareMesh(n, n)

# Reference pressure (macroscale)
C = float(1.0)

rho = Constant(1.0)
eta = Constant(1.0)

U0 = Constant(10.0)
U  = as_vector((U0, Constant(0.0)))  # slide in +x

# affine periodic jumps for physical pressure P
Ax = Constant(10.0)
Ay = Constant(-2.0)

# -----------------------------
# Film thickness on parent mesh
# -----------------------------
x, y = SpatialCoordinate(mesh)
epsH = Constant(0.2)
H_parent = Constant(0.5) + epsH*(cos(2*pi*x) + cos(2*pi*y))

# Cut threshold
hmin = Constant(0.2)

# -----------------------------
# Mark "fluid" cells on parent mesh
# fluid if H(cell midpoint) > hmin
# -----------------------------
tdim = mesh.topology().dim()
cell_mark = MeshFunction("size_t", mesh, tdim, 0)

H_expr = H_parent  # UFL expression on parent mesh
hmin_val = float(hmin)

for c in cells(mesh):
    mp = c.midpoint()
    # Evaluate H at midpoint (safe for this analytic H)
    Hmp = float(H_expr(Point(mp.x(), mp.y())))
    if Hmp > hmin_val:
        cell_mark[c] = 1  # fluid

# Guard: ensure we have some fluid cells
nfluid = sum(1 for c in cells(mesh) if cell_mark[c] == 1)
if nfluid == 0:
    raise RuntimeError("No fluid cells selected (H>hmin everywhere false). Lower hmin or change H.")

# -----------------------------
# Build SubMesh: fluid domain only
# -----------------------------
fluid_mesh = SubMesh(mesh, cell_mark, 1)

# Recreate coordinates/film thickness on fluid mesh
xf, yf = SpatialCoordinate(fluid_mesh)
H = Constant(0.5) + epsH*(cos(2*pi*xf) + cos(2*pi*yf))

# -----------------------------
# Reynolds coefficients (NO MASK)
# -----------------------------
K = rho*H**3/(12.0*eta)
F_adv = rho*(U*H/2.0)

# -----------------------------
# Cavitation penalty
# -----------------------------
gamma = Constant(1e4)
eps_smooth = Constant(1e-6)

def smooth_pos(z, eps):
    return 0.5*(z + sqrt(z*z + eps*eps))

# -----------------------------
# Periodic BC on fluid mesh
# (map only outer boundaries; internal boundary won't satisfy near(x,0/1) etc.)
# -----------------------------
class PeriodicBoundary(SubDomain):
    def inside(self, X, on_boundary):
        # Identify "master" boundary as x=0 or y=0 on the OUTER boundary
        # Exclude corners that are mapped twice.
        return on_boundary and (
            (near(X[0], 0.0) or near(X[1], 0.0)) and
            (not ((near(X[0], 1.0) and near(X[1], 0.0)) or
                  (near(X[0], 0.0) and near(X[1], 1.0)) or
                  (near(X[0], 1.0) and near(X[1], 1.0))))
        )

    def map(self, X, Y):
        if near(X[0], 1.0) and near(X[1], 1.0):
            Y[0] = X[0] - 1.0
            Y[1] = X[1] - 1.0
        elif near(X[0], 1.0):
            Y[0] = X[0] - 1.0
            Y[1] = X[1]
        else:
            Y[0] = X[0]
            Y[1] = X[1] - 1.0

pbc = PeriodicBoundary()
Vper  = FunctionSpace(fluid_mesh, "CG", 1, constrained_domain=pbc)
Vfull = FunctionSpace(fluid_mesh, "CG", 1)

# -----------------------------
# Affine ramp for jumps
# -----------------------------
phi = Ax*xf + Ay*yf

w = Function(Vper, name="w")
v = TestFunction(Vper)
dw = TrialFunction(Vper)

P = w + phi
negP_pos = smooth_pos(-P, eps_smooth)

# Weak form:
#   div(F_adv - K grad(P)) = 0
# with natural no-flux on *all* boundaries not handled by periodic constraints
R_reynolds = inner(F_adv - K*grad(P), grad(v))*dx
R_cav      = gamma * negP_pos * v * dx
R = R_reynolds + R_cav
J = derivative(R, w, dw)

# -----------------------------
# Point constraint for physical pressure at (0,0)
# (same logic as your masked version)
# -----------------------------
Axv = float(Ax)
Ayv = float(Ay)

p_ref0 = C - 0.5 * (Axv + Ayv)
min_delta = p_ref0 + min(0.0, Axv, Ayv, Ayv + Axv)
p_ref  = p_ref0 if (min_delta) >= 0.0 else (p_ref0 - min_delta)

bc_pin = DirichletBC(
    Vper,
    Constant(p_ref),
    "near(x[0],0.0) && near(x[1],0.0)",
    method="pointwise"
)

print(f"[CUT] Parent fluid cells: {nfluid} / {mesh.num_cells()}")
print(f"[CUT] Fluid mesh cells: {fluid_mesh.num_cells()}, vertices: {fluid_mesh.num_vertices()}")
print(f"[CUT] p_ref0={float(p_ref0):.6e}, p_ref={float(p_ref):.6e}")

# -----------------------------
# Nonlinear solve (Newton)
# -----------------------------
problem = NonlinearVariationalProblem(R, w, bcs=[bc_pin], J=J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["relative_tolerance"] = 1e-8
prm["newton_solver"]["maximum_iterations"] = 200
prm["newton_solver"]["report"] = True
prm["newton_solver"]["error_on_nonconvergence"] = False
prm["newton_solver"]["relaxation_parameter"] = 0.6
prm["newton_solver"]["linear_solver"] = "mumps"

w.vector().zero()
solver.solve()

# -----------------------------
# Export fields on fluid mesh
# -----------------------------
Pfull = project(P, Vfull)
Pfull.rename("P", "pressure_cutmesh")
File("pressure_affine_periodic_penalty_cav_cutmesh.pvd") << Pfull
print("\nWrote: pressure_affine_periodic_penalty_cav_cutmesh.pvd")

hfull = project(H, Vfull)
hfull.rename("H", "film_thickness")
File("h_affine_periodic_penalty_cav_cutmesh.pvd") << hfull
print("Wrote: h_affine_periodic_penalty_cav_cutmesh.pvd")

# Also export parent-cell marker (for visualising the cut region)
Vdg0_parent = FunctionSpace(mesh, "DG", 0)
cell_marker_fn = Function(Vdg0_parent, name="fluid_cell_marker")
marker_vals = cell_marker_fn.vector().get_local()
# DG0 dofs correspond to cells in order
marker_vals[:] = np.array([cell_mark[c] for c in cells(mesh)], dtype=float)
cell_marker_fn.vector().set_local(marker_vals)
cell_marker_fn.vector().apply("insert")
File("fluid_cell_marker_parent.pvd") << cell_marker_fn
print("Wrote: fluid_cell_marker_parent.pvd")

# -----------------------------
# Diagnostics: check shifted periodic jumps (sampling)
# WARNING: on cut mesh, some sample points may lie outside fluid domain.
# We'll sample a few points that are likely in fluid (adjust if needed).
# -----------------------------
def safe_eval(fun, xx, yy):
    try:
        return float(fun(Point(float(xx), float(yy))))
    except RuntimeError:
        return np.nan

xs = np.linspace(0.1, 0.9, 5)
ys = np.linspace(0.1, 0.9, 5)

print("\n[CUT] Check top-bottom jump P(x,1)-P(x,0) ~ Ay (NaN => point not in fluid mesh):")
for xx in xs:
    top = safe_eval(Pfull, xx, 1.0)
    bot = safe_eval(Pfull, xx, 0.0)
    print(f"x={xx:.2f}: {top - bot: .6e} (target {float(Ay): .6e})")

print("\n[CUT] Check left-right jump P(1,y)-P(0,y) ~ Ax:")
for yy in ys:
    right = safe_eval(Pfull, 1.0, yy)
    left  = safe_eval(Pfull, 0.0, yy)
    print(f"y={yy:.2f}: {right - left: .6e} (target {float(Ax): .6e})")

print("[CUT] Pinned value P(0,0) =", safe_eval(Pfull, 0.0, 0.0), " target =", float(p_ref))

pst = assemble(Pfull*dx)
print(f"[CUT] Mean pressure over FLUID MESH: {pst:.6e}")
