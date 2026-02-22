"""
Shifted periodic BCs with Reynolds
Penalty cavitation
Point constraint
"""

from dolfin import *
import numpy as np

# -----------------------------
# Mesh / params
# -----------------------------
n = 100
mesh = UnitSquareMesh(n, n)

#Reference pressure
C = float(1.0)  # your base reference level (like P or p0)

rho = Constant(1.0)
eta = Constant(1.0)

U0 = Constant(1.0)
U  = as_vector((U0, Constant(0.0)))  # slide in +x

# affine periodic jumps for physical pressure P
Ax = Constant(1.0)     # P(0,y) = P(1,y) + Ax
Ay = Constant(-2.0)    # P(x,1) = P(x,0) + Ay

# film thickness (periodic, positive)
x, y = SpatialCoordinate(mesh)
epsH = Constant(0.2)
H = Constant(0.5) + epsH*(cos(2*2*pi*x) + cos(2*2*pi*y))
K = rho*H**3/(12.0*eta)
F_adv = rho*(U*H/2.0)

# penalty cavitation parameters
gamma = Constant(1e4)   # penalty strength (increase to enforce P>=0 harder)
eps_smooth = Constant(1e-6)  # smoothing for positive-part approx

# -----------------------------
# Periodic BC for w
# -----------------------------
class PeriodicBoundary(SubDomain):
    def inside(self, X, on_boundary):
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
Vper  = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
Vfull = FunctionSpace(mesh, "CG", 1)  # for exporting P

# -----------------------------
# Affine ramp phi for jumps
# -----------------------------
phi = Ax*x + Ay*y   # gives desired jumps when w is periodic

# Unknown in periodic space
w = Function(Vper, name="w")
v = TestFunction(Vper)
dw = TrialFunction(Vper)

# Physical pressure (UFL expression)
P = w + phi

# Smooth positive part: pos(z) ~ 0.5*(z + sqrt(z^2 + eps^2))
def smooth_pos(z, eps):
    return 0.5*(z + sqrt(z*z + eps*eps))

# Cavitation penalty uses pos(-P)
negP_pos = smooth_pos(-P, eps_smooth)

# -----------------------------
# Weak form (periodic => no boundary terms)
# Reynolds: div(F_adv - K*grad(P)) = 0
# Weak: ∫ (F_adv - K*grad(P)) · grad(v) dx = 0
#
# Add penalty: + ∫ gamma * pos(-P) * v dx
# -----------------------------
R_reynolds = inner(F_adv - K*grad(P), grad(v))*dx
R_cav      = gamma * negP_pos * v * dx

R = R_reynolds + R_cav

J = derivative(R, w, dw)

# # -----------------------------
# # Gauge fixing: pin physical pressure at a point, e.g. P(0,0)=0
# # => w(0,0) = -phi(0,0) = -Ax
# # -----------------------------
# bc_pin = DirichletBC(Vper, Constant(-float(Ax)),
#                      "near(x[0],0.0) && near(x[1],0.0)",
#                      method="pointwise")

# -----------------------------
# Point constraint value for physical pressure
# -----------------------------

Axv = float(Ax)
Ayv = float(Ay)

# corner pressures implied by pinning P(0,0)=p_ref:
# P00 = p_ref
# P10 = p_ref - Ax
# P01 = p_ref + Ay
# P11 = p_ref + Ay - Ax


p_ref0 = C - 0.5 * (Axv + Ayv)
min_delta = p_ref0 + min(0.0, Axv, Ayv, Ayv + Axv)
# print(f'min delta {min_delta}')

p_ref  = p_ref0 if (min_delta) >= 0.0 else (p_ref0 - min_delta)

bc_pin = DirichletBC(
    Vper,
    Constant(p_ref),
    "near(x[0],0.0) && near(x[1],0.0)",
    method="pointwise"
)

print(f'p_ref_0 {float(p_ref0)}')
print(f'p_ref {float(p_ref)}')
# -----------------------------
# Gauge fixing: pin physical pressure at a point
# P(0,0)=C  =>  w(0,0) = C - phi(0,0) = C - Ax
# -----------------------------
bc_pin = DirichletBC(
    Vper,
    p_ref,
    "near(x[0],0.0) && near(x[1],0.0)",
    method="pointwise"
)


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
# linear solver choice
prm["newton_solver"]["linear_solver"] = "mumps"  # if available; otherwise try "petsc"

# initial guess (optional): start from 0
w.vector().zero()
solver.solve()

# -----------------------------
# Export physical pressure
# -----------------------------
Pfull = project(P, Vfull)
Pfull.rename("P", "pressure")
File("pressure_affine_periodic_penalty_cav.pvd") << Pfull
print("\nWrote: pressure_affine_periodic_penalty_cav.pvd")

hfull = project(H, Vfull)
hfull.rename("H", "film thickness")
File("h_affine_periodic_penalty_cav.pvd") << hfull

# -----------------------------
# Diagnostics: check jumps and cavitation extent
# -----------------------------
def P_eval(xx, yy):
    return Pfull(Point(float(xx), float(yy)))

xs = np.linspace(0.1, 0.9, 5)
ys = np.linspace(0.1, 0.9, 5)

print("\nCheck top-bottom jump P(x,1)-P(x,0) ~ Ay:")
for xx in xs:
    print(f"x={xx:.2f}: {P_eval(xx,1.0) - P_eval(xx,0.0): .6e} (target {float(Ay): .6e})")

print("\nCheck left-right jump P(1,y)-P(0,y) ~ Ax:")
for yy in ys:
    print(f"y={yy:.2f}: {P_eval(1.0,yy) - P_eval(0.0,yy): .6e} (target {float(Ax): .6e})")

print("Pinned value P(0,0) =", Pfull(Point(0.0, 0.0)), " target =", float(p_ref))

print("Corner pressures implied by pin:")
print("P00", p_ref)
print("P10", p_ref + Axv)
print("P01", p_ref + Ayv)
print("P11", p_ref + Axv + Ayv)
print("min corner", min(p_ref, p_ref+Axv, p_ref+Ayv, p_ref+Axv+Ayv))


# Estimate minimum pressure (will usually be slightly negative unless gamma very large)
# Sample a grid:
grid = np.linspace(0, 1, 41)
pmin = 1e100
for xx in grid:
    for yy in grid:
        pmin = min(pmin, P_eval(xx, yy))
print(f"\nApprox min(P) on sampled grid: {pmin:.6e}")

#Homogenise pressure
pst = assemble(Pfull*dx)
print(f'Homogenised pressure {pst}')