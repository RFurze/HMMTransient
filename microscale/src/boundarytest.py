from math import pi
import ngsolve
import logging
from ngsolve import *
from ngsolve.solvers import *
import numpy as np
import time
import sys
import io
from contextlib import contextmanager
import gc


from microscale.src.functions.micro_meshing import GenerateSquareMesh_Scaled, GenerateSquareMesh_Scaled_Adaptive


mesh = GenerateSquareMesh_Scaled_Adaptive(30, 1e-6, 1e-6)


V = H1(mesh, order=2, dirichlet="left|right|top|bottom")
u, v = V.TrialFunction(), V.TestFunction()

# define a coefficient function to apply a fixed value on each boundary
# Define the boundary conditions
bdry_values = {'bottom': (1), 'top': (0), 'left': (2), 'right': (3)}
cf = mesh.BoundaryCF(bdry_values, default=(0))

gfu = GridFunction(V)
gfu.Set(cf, definedon=mesh.Boundaries("bottom|top|left|right"))

vtk = VTKOutput(mesh, coefs=[gfu], names=["gfu"], filename="gfu", subdivision=1)
vtk.Do()
