import pytest

import numpy as np

import fgspectra.power as fgp
import fgspectra.frequency as fgf
import fgspectra.cross as fgc


ell = np.arange(2, 5000)
nu = np.linspace(80.0, 250.0, 100)


def test_all():
    cl = fgp.PowerLaw()
    sed = fgf.PowerLawSED()
    cross = fgc.FactorizedCrossSpectrum(sed, cl)

    print(cross.parameters)
