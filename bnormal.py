#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, \
    MeanSquaredCurvature, LpCurveCurvature

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--NF", type=int, default=10)
parser.add_argument("--NSEG", type=int, default=128)
parser.add_argument("--NPHI", type=int, default=64)
parser.add_argument("--NTHETA", type=int, default=64)
parser.add_argument("--MODE", type=int, default=0)
args = parser.parse_args()
# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = args.NF

# Weight on the curve lengths in the objective function:
LENGTH_WEIGHT = 1e-6

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
DISTANCE_THRESHOLD = 0.1
DISTANCE_WEIGHT = 10

NITER = 100

# File for the desired boundary magnetic surface:
filename = 'input.LandremanPaul2021_QA'

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = args.NPHI
ntheta = args.NTHETA
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=args.NSEG)
base_currents = [Current(1e5) for i in range(ncoils)]

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]

# Define the objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jdist = MinimumDistance(curves, DISTANCE_THRESHOLD)

if args.MODE == 0:
    JF = Jf
elif args.MODE == 1:
    JF = Jf + LENGTH_WEIGHT * sum(Jls)
elif args.MODE == 2:
    JF = Jf + LENGTH_WEIGHT * sum(Jls) + DISTANCE_WEIGHT * Jdist


# execute once to force JIT by JAX
JF.J()
JF.dJ()
dofs = JF.x


################################################################################
### Run 100 random function and gradient evaluations ###########################
################################################################################


def f(dofs):
    JF.x = dofs
    return JF.J()

def fg(dofs):
    JF.x = dofs
    return JF.J(), JF.dJ()

import time
tic = time.time()
for i in range(NITER):
    bs.clear_cached_properties()
    Jf.J()
toc = time.time()
print(f"Mode={args.MODE}, OMP={os.getenv('OMP_NUM_THREADS')}")
print(f"Time for {NITER} evaluations of ∫(B·n)²ds : {toc-tic:.2f}")

tic = time.time()
for i in range(NITER):
    # fun(dofs + 1e-2 * np.random.standard_normal(size=dofs.shape))
    fg(dofs + 1e-2 * np.random.standard_normal(size=dofs.shape))
toc = time.time()
print(f"Time for {NITER} iterations of a gradient based algorithm: {toc-tic:.2f}")
