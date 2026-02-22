"""
Shifted periodic BCs with Reynolds
Penalty cavitation
Point constraint

Modified: add immersed / masked Reynolds physics based on film thickness H, following the
masking logic in mixed_testing_dummy2.py (NGSolve version).

Key idea:
- Build a smooth (or sharp) mask m(H) ~ 1 in fluid region (H > hmin) and ~ eps_solid in solid/contact region (H < hmin)
- Multiply mobility K and advection flux F_adv by m(H) so the PDE is effectively "turned off" in solid regions
- Optionally also weight cavitation penalty by the same mask so cavitation is only enforced where fluid exists
"""

from dolfin import *
import numpy as np

# -----------------------------
# Mesh / params
# -----------------------------
n = 200
mesh = UnitSquareMesh(n, n)

# Reference pressure (macroscale)
C = float(1.0)  # base reference level

rho = Constant(1.0)
eta = Constant(1.0)

U0 = Constant(10.0)
U  = as_vector((U0, Constant(0.0)))  # slide in +x

# affine periodic jumps for physical pressure P
Ax = Constant(10.0)     # P(0,y) = P(1,y) + Ax
Ay = Constant(-2.0)    # P(x,1) = P(x,0) + Ay

# -----------------------------
# Film thickness (periodic, positive)
# -----------------------------
x, y = SpatialCoordinate(mesh)
epsH = Constant(0.2)
H = Constant(0.5) + epsH*(cos(2*pi*x) + cos(2*pi*y))

# -----------------------------
# Masking / immersed "no-flow" region definition
# (matches spirit of mixed_testing_dummy2.py)
# -----------------------------
hmin = Constant(0.002)        # minimum film thickness defining "fluid" vs "solid" region
delta_h = Constant(0.0)      # transition half-width; 0 => sharp mask
eps_solid = Constant(1e-8)   # residual mobility inside solid/contact region (avoid singular matrices)

def smooth_levelset_mask(phi, delta, mask_core, mask_fluid):
    """
    UFL version of the NGSolve mask:
      phi = H - hmin

      mask = mask_fluid               if phi > +delta
           = mask_core                if phi < -delta
           = linear transition        otherwise
    """
    # sharp mask if delta == 0 (or extremely small)
    # NOTE: we branch in Python on the *float* value to avoid division by zero in UFL.
    if float(delta) == 0.0:
        return conditional(gt(phi, 0.0), mask_fluid, mask_core)

    t = (phi + delta) / (2.0*delta)  # maps [-delta, +delta] -> [0, 1]
    mask_transition = mask_core + (mask_fluid - mask_core)*t

    return conditional(gt(phi - delta, 0.0),
                       mask_fluid,
                       conditional(gt((-delta) - phi, 0.0),
                                   mask_core,
                                   mask_transition))

phi_mask = H - hmin
chi_fluid = conditional(gt(phi_mask, 0.0), Constant(1.0), Constant(0.0))  # for post-processing only
mask = smooth_levelset_mask(phi_mask, delta_h, eps_solid, Constant(1.0))

# -----------------------------
# Reynolds coefficients (unchanged physics, but masked)
# -----------------------------
K_base = rho*H**3/(12.0*eta)
F_adv_base = rho*(U*H/2.0)

# "Immersed" coefficients
K = K_base * mask
F_adv = F_adv_base * mask

# -----------------------------
# Cavitation penalty parameters
# -----------------------------
gamma = Constant(1e8)        # penalty strength (increase to enforce P>=0 harder)
eps_smooth = Constant(1e-6)  # smoothing for positive-part approximation

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
Vfull = FunctionSpace(mesh, "CG", 1)  # for pressure/film exports
# Discontinuous space for exporting sharp indicator fields without CG ringing/overshoot
Vdg0  = FunctionSpace(mesh, "DG", 0)

# -----------------------------
# Affine ramp phi for jumps
# -----------------------------
phi = Ax*x + Ay*y   # desired jumps when w is periodic

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
# Masking:
# - F_adv and K are already multiplied by mask
# - cavitation penalty is also weighted by mask, so solid/contact regions do not
#   "care" about P>=0 (pressure there is not physical)
# -----------------------------
R_reynolds = inner(F_adv - K*grad(P), grad(v))*dx
R_cav      = gamma * mask * negP_pos * v * dx

R = R_reynolds + R_cav
J = derivative(R, w, dw)

# -----------------------------
# Point constraint value for physical pressure (unchanged)
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

print(f'p_ref_0 {float(p_ref0)}')
print(f'p_ref {float(p_ref)}')

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
prm["newton_solver"]["relaxation_parameter"] = 0.2
prm["newton_solver"]["linear_solver"] = "mumps"  # if available; otherwise try "petsc"

# initial guess (optional): start from 0
w.vector().zero()
solver.solve()

# -----------------------------
# Export physical pressure + mask fields
# -----------------------------
# Unmasked (physical) pressure
Pfull = project(P, Vfull)
Pfull.rename("P", "pressure_unmasked")
File("pressure_unmasked_affine_periodic_penalty_cav_masked.pvd") << Pfull
print("\nWrote: pressure_unmasked_affine_periodic_penalty_cav_masked.pvd")

# Masked pressure with NaNs outside the fluid region (ParaView can treat NaN as blank)
# NOTE: dolfin.interpolate cannot take a raw UFL Conditional, so first project chi into DG0,
# then interpolate that *Function* to CG for nodal masking.
chi_dg0   = project(chi_fluid, Vdg0)       # bounded cellwise in [0,1]
chi_nodal = interpolate(chi_dg0, Vfull)    # nodal values for masking

p_arr   = Pfull.vector().get_local()
chi_arr = chi_nodal.vector().get_local()

Pmasked = Function(Vfull)
arr_nan = p_arr.copy()
arr_nan[chi_arr <= 0.5] = np.nan
Pmasked.vector().set_local(arr_nan)
Pmasked.vector().apply("insert")
Pmasked.rename("P", "pressure")
File("pressure_affine_periodic_penalty_cav_masked.pvd") << Pmasked
print("Wrote: pressure_affine_periodic_penalty_cav_masked.pvd")

# Film thickness
hfull = project(H, Vfull)
hfull.rename("H", "film thickness")
File("h_affine_periodic_penalty_cav_masked.pvd") << hfull

# Export indicators in DG0 to avoid CG projection overshoot near sharp interfaces
maskfull = project(mask, Vdg0)
maskfull.rename("mask", "immersed mask")
File("mask_affine_periodic_penalty_cav_masked.pvd") << maskfull

chifull = chi_dg0
chifull.rename("chi_fluid", "fluid indicator")
File("chi_fluid_affine_periodic_penalty_cav_masked.pvd") << chifull

# -----------------------------
# Diagnostics: check jumps and cavitation extent (unchanged)
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
grid = np.linspace(0, 1, 41)
pmin = 1e100
for xx in grid:
    for yy in grid:
        pmin = min(pmin, P_eval(xx, yy))
print(f"\nApprox min(P) on sampled grid: {pmin:.6e}")

# Homogenise pressure
pst = assemble(Pfull*dx)
print(f'Homogenised pressure {pst}')

# Masked (fluid-only) average pressure (optional diagnostic)
pst_fluid = assemble(chi_fluid * Pfull * dx) / max(assemble(chi_fluid*dx), 1e-30)
print(f'Fluid-only mean pressure {pst_fluid}')

print("\nMasking params:")
print(f"  hmin={float(hmin)}, delta_h={float(delta_h)}, eps_solid={float(eps_solid)}")
