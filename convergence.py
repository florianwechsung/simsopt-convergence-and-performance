#!/usr/bin/env python
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries

ncoils = 4
R0 = 1.0
R1 = 0.5
order = 10
filename = 'input.LandremanPaul2021_QA'


def get_jflux(nphi=32, ntheta=32, nseg=100, export=False):
    # Initialize the boundary magnetic surface:
    s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=nseg)
    base_currents = [Current(1e5) for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(s, bs)
    np.random.seed(0)
    Jf.x = Jf.x + 0.01 * np.random.standard_normal(size=Jf.x.shape)
    if export:
        curves_to_vtk([c.curve for c in coils], "/tmp/coils_convergence")
        for i, c in enumerate(coils):
            np.savetxt(f"coil_{i}_xyz.txt", c.curve.gamma())
    return Jf.J()

acc = get_jflux(nphi=1024, ntheta=1024, nseg=1024, export=True)
print("Accurate solution=", acc)

for phitheta in [32, 64, 128, 256]:
    for nseg in [32, 64, 128, 256]:
        print(f"{abs(acc-get_jflux(nphi=phitheta, ntheta=phitheta, nseg=nseg)):.3e}", end=" ")
    print("")
