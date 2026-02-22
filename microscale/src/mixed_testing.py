from ngsolve import *
from netgen.geom2d import unit_square
import math

# ----------------------------
# Mesh / Space
# ----------------------------
mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
fes  = H1(mesh, order=2, dirichlet="left|right|bottom|top")
p    = GridFunction(fes)   # total pressure (for this demo)
v    = fes.TestFunction()
u    = fes.TrialFunction()

# ----------------------------
# Parameters (toy values)
# ----------------------------
eta   = 1.0          # viscosity (not used explicitly here)
H     = 1.0e-6       # nominal film thickness offset
k1    = 1.0e-12      # compliance: delta = k1 * p
hmin  = 0.2e-6       # minimum film thickness

eps_beta = 1e-6      # tiny mobility in contact (prevents singularity)
alpha_c  = 1e12      # contact penalty strength (tune!)

# (Optional) cap pressures / targets
pmax = 1e9

# ----------------------------
# Topography (example)
# Replace with your measured / generated roughness field.
# ----------------------------
x, y = symbols("x y")
ht = 0.15e-6 * sin(2*math.pi*x/0.25) * sin(2*math.pi*y/0.25)

# base gap without deformation
hbase = 2*ht + H

# Target pressure that would enforce h = hmin if in contact:
# hbase + k1*p = hmin  =>  p = (hmin - hbase)/k1
# Only meaningful where hmin > hbase (otherwise target negative -> no contact)
p_target = IfPos(hmin - hbase, (hmin - hbase)/k1, 0.0)
p_target = Min(p_target, pmax)

# ----------------------------
# Helper: compute gap from current pressure
# ----------------------------
def gap_from_pressure(p_gf):
    # h = hbase + k1*p
    return hbase + k1*p_gf

# ----------------------------
# Iteration: Active-set (contact mask) + blended solve
# ----------------------------
maxit = 50
tol_area = 0.0  # use e.g. 1e-3*domain area if you want a tolerance
prev_Ac = None

# initialise with zero pressure
p.vec[:] = 0.0

for it in range(maxit):
    # Compute current gap using previous iterate
    hcur = gap_from_pressure(p)

    # Active set: m=1 where gap < hmin (contact), else 0
    # You can also define it via pc>0 if you later introduce a contact law.
    m = IfPos(hmin - hcur, 1.0, 0.0)

    # Mobility modifier (fluid region has mobility ~1, contact region ~eps_beta)
    beta = (1.0 - m) + eps_beta*m

    # Reynolds-like operator coefficient ~ beta*h^3
    acoef = beta * hcur**3

    # OPTIONAL: simple "source" term (placeholder for wedge / Couette forcing)
    # In a real Reynolds equation you’d include convective terms / RHS based on U, h gradients, etc.
    f = 0.0

    # Build bilinear form:
    # Fluid region (1-m): diffusion with acoef
    # Contact region (m): penalty to enforce p ≈ p_target
    a = BilinearForm(fes, symmetric=True)
    a += (1.0-m) * acoef * grad(u) * grad(v) * dx
    a += m * alpha_c * u * v * dx

    # RHS:
    l = LinearForm(fes)
    l += (1.0-m) * f * v * dx
    l += m * alpha_c * p_target * v * dx

    # Assemble and solve
    a.Assemble()
    l.Assemble()

    # Preconditioner: you can use local, multigrid, or direct depending on problem size
    pre = Preconditioner(a, "local")
    pre.Update()

    # Solve (CG is fine for symmetric positive definite form)
    sol = p.vec.CreateVector()
    sol.data = CGSolver(a.mat, pre.mat, maxsteps=500, precision=1e-10) * l.vec
    p.vec.data = sol

    # ----------------------------
    # Postprocess: contact area and "contact load"
    # ----------------------------
    # Contact area:
    Ac = Integrate(m, mesh)

    # If you interpret contact pressure as the penalty traction:
    # pc ≈ alpha_c*(p - p_target) ??? Not physically meaningful.
    #
    # Better: use your own Hertz-like contact pressure law once you have the gap:
    # pc_hertz = K * <hmin - h>^(3/2) (optionally smoothed)
    #
    # For now: compute an example Hertz-like contact pressure field from overclosure:
    E_eff = 1.0e9       # placeholder effective modulus
    R_eff = 1.0e-6      # placeholder effective radius
    K_hz  = (4.0/3.0) * E_eff * math.sqrt(R_eff)

    over = IfPos(hmin - hcur, hmin - hcur, 0.0)
    pc_hertz = K_hz * over**(1.5)

    Wc = Integrate(pc_hertz, mesh)  # asperity load (normal force)

    # Convergence check based on contact area stabilising
    if prev_Ac is not None:
        if abs(Ac - prev_Ac) <= max(tol_area, 1e-14):
            print(f"[it={it}] Converged by area: Ac={Ac:.6e}, Wc={Wc:.6e}")
            break

    prev_Ac = Ac
    print(f"[it={it}] Ac={Ac:.6e}, Wc={Wc:.6e}")

# At this point:
# - p is your blended (fluid/contact) pressure field from the active-set solve
# - m gives the final contact set
# - Ac and Wc are contact area and Hertz-style asperity load (from gap)
