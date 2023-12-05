# %% [markdown]
# # 3D Code for  isothermal finite hyperelasticity
#
# Sinusoidal Simple shear
#
# Basic units:
# Length: mm
# Mass: kg
# Time:  s
# Derived units:
# Force: milliNewtons
# Stress: kPa
#
# Eric Stewart and Lallit Anand
# ericstew@mit.edu and anand@mit.edu
#
# Converted to FEniCSx by Jorge Nin
# jorgenin@mit.edu
# September 2023
#
#

# %%
import numpy as np
import dolfinx

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import Constant, dirichletbc, Function, FunctionSpace, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
import ufl
from ufl import (
    TestFunctions,
    TrialFunction,
    Identity,
    grad,
    det,
    div,
    dev,
    inv,
    tr,
    sqrt,
    conditional,
    gt,
    dx,
    inner,
    derivative,
    dot,
    ln,
    split,
)
from datetime import datetime
from dolfinx.plot import vtk_mesh

import pyvista

pyvista.set_jupyter_backend("client")
## Define temporal parameters

# %% [markdown]
# ### Set level of detail for log messages (integer)
# Guide: \
# CRITICAL  = 50  errors that may lead to data corruption \
# ERROR     = 40  things that HAVE gone wrong \
# WARNING   = 30  things that MAY go wrong later \
# INFO      = 20  information of general interest (includes solver info) \
# PROGRESS  = 16  what's happening (broadly) \
# TRACE     = 13  what's happening (in detail) \
# DBG       = 10  sundry
#

# %%
log.set_log_level(log.LogLevel.WARNING)

# %% [markdown]
# # Define Geometry

# %%
from types import CellType


L = 1.0  # mm
domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, L, L * 20]], [4, 4, 20])
x = ufl.SpatialCoordinate(domain)

# %% [markdown]
# ### Visualize Gemometry


# %% [markdown]
# ## Functions for finding Differnent Areas


# %%
def xBot(x):
    return np.isclose(x[0], 0)


def xTop(x):
    return np.isclose(x[0], L)


def yBot(x):
    return np.isclose(x[1], 0)


def yTop(x):
    return np.isclose(x[1], L)


def zBot(x):
    return np.isclose(x[2], 0)


def zTop(x):
    return np.isclose(x[2], L * 20)


# %%
boundaries = [(1, xBot), (2, xTop), (3, yBot), (4, yTop), (5, zBot), (6, zTop)]

facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for marker, locator in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(
    domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
)

ds = ufl.Measure(
    "ds", domain=domain, subdomain_data=facet_tag, metadata={"quadrature_degree": 4}
)
n = ufl.FacetNormal(domain)

# %% [markdown]
# ## MATERIAL PARAMETERS
# Arruda-Boyce Model

# %%
Gshear_0 = Constant(domain, PETSc.ScalarType(280.0))  # Ground state shear modulus
lambdaL = Constant(domain, PETSc.ScalarType(5.12))  # Locking stretch
Kbulk = Constant(domain, PETSc.ScalarType(1000.0 * Gshear_0))

# %% [markdown]
# ## Simulation Control

print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time
startTime = datetime.now()

# %%
t = 0.0  # start time (s)
dispTot = -2.5  # mm
Ttot = 1
numSteps = 100
# Total time
dt = Ttot / numSteps  # (fixed) step size


def distRamp(t):
    return dispTot * t / Ttot


# %% [markdown]
# ## Function Spaces

# %%


U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)  # For displacement
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # For  pressure
TH = ufl.MixedElement([U2, P1])  # Taylor-Hood style mixed element
ME = FunctionSpace(domain, TH)  # Total space for all DOFs

# %%
w = Function(ME)
u, p = split(w)
w_old = Function(ME)
u_old, p_old = split(w_old)

u_test, p_test = TestFunctions(ME)
dw = TrialFunction(ME)

# %% [markdown]
# ## SubRoutine


# %%
def F_calc(u):
    Id = Identity(3)
    F = Id + grad(u)
    return F


def lambdaBar_calc(u):
    F = F_calc(u)
    C = F.T * F
    J = det(F)
    Cdis = J ** (-2 / 3) * C
    I1 = tr(Cdis)
    lambdaBar = sqrt(I1 / 3.0)
    return lambdaBar


def zeta_calc(u):
    lambdaBar = lambdaBar_calc(u)
    # Use Pade approximation of Langevin inverse
    z = lambdaBar / lambdaL
    z = conditional(gt(z, 0.95), 0.95, z)  # Keep simulation from blowing up
    beta = z * (3.0 - z**2.0) / (1.0 - z**2.0)
    zeta = (lambdaL / (3 * lambdaBar)) * beta
    return zeta


# Generalized shear modulus for Arruda-Boyce model
def Gshear_AB_calc(u):
    zeta = zeta_calc(u)
    Gshear = Gshear_0 * zeta
    return Gshear


# ---------------------------------------------
# Subroutine for calculating the Cauchy stress
# ---------------------------------------------
def T_calc(u, p):
    Id = Identity(3)
    F = F_calc(u)
    J = det(F)
    B = F * F.T
    Bdis = J ** (-2 / 3) * B
    Gshear = Gshear_AB_calc(u)
    T = (1 / J) * Gshear * dev(Bdis) - p * Id
    return T


# ----------------------------------------------
# Subroutine for calculating the Piola  stress
# ----------------------------------------------
def Piola_calc(u, p):
    Id = Identity(3)
    F = F_calc(u)
    J = det(F)
    #
    T = T_calc(u, p)
    #
    Tmat = J * T * inv(F.T)
    return Tmat


# %%
F = F_calc(u)
J = det(F)
lambdaBar = lambdaBar_calc(u)

# Piola stress
Tmat = Piola_calc(u, p)

# %% [markdown]
# ## WEAK FORMS

# %%
dxs = dx(metadata={"quadrature_degree": 4})

# %%
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Coupling pressure (test fxn: p)

# The weak form for the equilibrium equation. No body force
Res_0 = inner(Tmat, grad(u_test)) * dxs

# The weak form for the pressure
fac_p = ln(J) / J
#
Res_1 = dot((p / Kbulk + fac_p), p_test) * dxs

# Total weak form
Res = Res_0 + Res_1

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

# %%
Time_cons = Constant(domain, PETSc.ScalarType(distRamp(0)))
ZeroValue = Constant(domain, PETSc.ScalarType([0]))
# U0, submap = ME.sub(0).sub(1).collapse()
# fixed_displacement = fem.Function(U0)
# fixed_displacement.interpolate(lambda x :   np.full(x.shape[1], distRamp(Time_cons)))


zBot_dofs = fem.locate_dofs_topological(ME.sub(0), facet_tag.dim, facet_tag.find(5))
zTop_dofsu1 = fem.locate_dofs_topological(
    ME.sub(0).sub(0), facet_tag.dim, facet_tag.find(6)
)
zTop_dofsu2 = fem.locate_dofs_topological(
    ME.sub(0).sub(1), facet_tag.dim, facet_tag.find(6)
)
zTop_dofsu3 = fem.locate_dofs_topological(
    ME.sub(0).sub(2), facet_tag.dim, facet_tag.find(6)
)


bcs_1 = dirichletbc(ZeroValue, zBot_dofs, ME.sub(0))  #  z bottom fixed
bcs_2 = dirichletbc(Time_cons, zTop_dofsu3, ME.sub(0).sub(2))  # u3 ramp - zTop
bcs_3 = dirichletbc(0.0, zTop_dofsu1, ME.sub(0).sub(0))  # u1 fixd  - zTop
#
bcs_4 = dirichletbc(0.0, zTop_dofsu2, ME.sub(0).sub(1))  # u2 fixed  -  zTop
bcs = [bcs_1, bcs_2, bcs_3, bcs_4]

# %% [markdown]
# ## Non Linear Variational

# %%
# Setting up visualziation
import pyvista
import matplotlib
import cmasher as cmr
import os

if not os.path.exists("results"):
    # Create a new directory because it does not exist
    os.makedirs("results")

if os.path.exists("results/3D_buckling_fixed.xdmf"):
    os.remove("results/3D_buckling_fixed.xdmf")
    os.remove("results/3D_buckling_fixed.h5")


# %%
## Functions for visualization


U1 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
V2 = fem.FunctionSpace(domain, U1)  # Vector function space
V1 = fem.FunctionSpace(domain, P1)  # Scalar function space

u_r = Function(V2)

u_r.name = "disp"

p_r = Function(V1)

p_r.name = "p"

J_vis = Function(V1)
J_vis.name = "J"
J_expr = Expression(J, V1.element.interpolation_points())


lambdaBar_Vis = Function(V1)
lambdaBar_Vis.name = "lambdaBar"
lambdaBar_expr = Expression(lambdaBar, V1.element.interpolation_points())


P11 = Function(V1)
P11.name = "P11"
P11_expr = Expression(Tmat[0, 0], V1.element.interpolation_points())
P22 = Function(V1)
P22.name = "P22"
P22_expr = Expression(Tmat[1, 1], V1.element.interpolation_points())
P33 = Function(V1)
P33.name = "P33"
P33_expr = Expression(Tmat[2, 2], V1.element.interpolation_points())

T = Tmat * F.T / J
T0 = T - (1 / 3) * tr(T) * Identity(3)
Mises = sqrt((3 / 2) * inner(T0, T0))
Mises_Vis = Function(V1, name="Mises")
Mises_expr = Expression(Mises, V1.element.interpolation_points())


def InterpAndSave(t, file):
    u_r.interpolate(w.sub(0))
    p_r.interpolate(w.sub(1))
    J_vis.interpolate(J_expr)
    P11.interpolate(P11_expr)
    P22.interpolate(P22_expr)
    P33.interpolate(P33_expr)
    lambdaBar_Vis.interpolate(lambdaBar_expr)
    Mises_Vis.interpolate(Mises_expr)

    file.write_function(u_r, t)
    file.write_function(p_r, t)
    file.write_function(J_vis, t)
    file.write_function(P11, t)
    file.write_function(P22, t)
    file.write_function(P33, t)
    file.write_function(lambdaBar_Vis, t)
    file.write_function(Mises_Vis, t)


pointForStress = [L / 2, L / 2, L * 20]
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pointForStress)
colliding_cells = dolfinx.geometry.compute_colliding_cells(
    domain, cell_candidates, pointForStress
)


Force = fem.form(P33 / (L * L) * ds(6))  # Piola stress


# %%


jit_options = {"cffi_extra_compile_args": ["-O3", "-ffast-math"]}

step = "Buckling"

problem = NonlinearProblem(Res, w, bcs, a, jit_options=jit_options)


solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.report = True

ksp = solver.krylov_solver
# ksp.setType(PETSc.KSP.Type.CG)
# pc = ksp.getPC()
# pc.setType(PETSc.PC.Type.HYPRE)
# pc.setHYPREType("boomeramg")

opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "cg"
# opts[f"{option_prefix}pc_type"] = "gamg"
# opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# opts[f"{option_prefix}ksp_max_it"] = 30
# ksp.setFromOptions()

# Initalize output array for tip displacement
totSteps = numSteps + 1
timeHist0 = np.zeros(shape=[totSteps])
timeHist1 = np.zeros(shape=[totSteps])

# Iinitialize a counter for reporting data
ii = 0

xdmf = XDMFFile(domain.comm, "results/3D_buckling_fixed.xdmf", "w")
xdmf.write_mesh(domain)


InterpAndSave(t, xdmf)


print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time
startTime = datetime.now()

while round(t + dt, 9) <= Ttot:
    # increment time
    t += dt
    # increment counter
    ii += 1

    # update time variables in time-dependent BCs
    Time_cons.value = distRamp(t)

    # Solve the problem
    try:
        (iter, converged) = solver.solve(w)
    except:  # Break the loop if solver fails
        print("Ended Early")
        break

    w.x.scatter_forward()
    # Write output to *.xdmf file
    # writeResults(t)
    # print(u0.x.array-w.x.array[dofs])
    # Update DOFs for next step
    w_old.x.array[:] = w.x.array
    # SAVING RESULT
    InterpAndSave(t, xdmf)
    # Visualizing GIF

    # Store  displacement at a particular point  at this time
    # timeHist0[ii] = w.sub(0).sub(2).eval(pointForStress,colliding_cells[0])[0]
    #

    # timeHist1[ii] =  domain.comm.gather(fem.assemble_scalar(Force))[0] #Convert from UFL operator to a number in milli Newtons

    # Print progress of calculation
    if ii % 1 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(
            "Step: {} | Simulation Time: {} s, Wallclock Time: {}".format(
                step, round(t, 4), current_time
            )
        )
        print("Iterations: {}".format(iter))
        print()


print("-----------------------------------------")
print("End computation")
# Report elapsed real time for the analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------------")
xdmf.close()

# %%


# %%
import matplotlib.pyplot as plt

font = {"size": 14}
plt.rc("font", **font)
# Get array of default plot colors
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

# # Only plot as far as we have time history data
# ind = np.argmax(timeHist0)
#
# Create array for plotting simulation data
# times=np.linspace(0, Ttot, num=totSteps)

#
fig = plt.figure()
# fig.set_size_inches(7,4)
ax = fig.gca()
# --------------------------------------------------------------
plt.plot(timeHist0, timeHist1, linewidth=2.0, color=colors[0], marker=".")
# -------------------------------------------------------------
# plt.xlim([1,8])
# plt.ylim([0,8])
plt.axis("tight")
plt.ylabel(r"$P_{33}$, kPa")
plt.xlabel(r"$u_3$ (mm)")
plt.grid(linestyle="--", linewidth=0.5, color="b")
# -------------------------------------------------------------
# ax.set_title("Stress dispalcement  curve", size=14, weight='normal')
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.show()

fig = plt.gcf()
fig.set_size_inches(7, 5)
plt.tight_layout()
plt.savefig("results/3D_columnn_buckling.png", dpi=600)
