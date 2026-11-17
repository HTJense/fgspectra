import pytest

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_SKIP = False
except Exception as e:
    JAX_SKIP = True

import fgspectra.power as fgp
import fgspectra.frequency as fgf
import fgspectra.cross as fgc


def test_float():
    ell = 5
    nu = 150.

    constant = fgf.ConstantSED()
    power_law = fgp.PowerLaw()
    cross = fgc.FactorizedCrossSpectrum(constant, power_law)
    cl_spec = cross(nu, nu, ell, alpha=1.0)

    assert isinstance(cl_spec, np.ndarray)
    assert cl_spec.ndim == 3
    assert cl_spec.shape[0] == 1
    assert cl_spec.shape[1] == 1
    assert cl_spec.shape[2] == 1


def test_numpy():
    ell = np.arange(2, 5000)
    nu = np.linspace(90.0, 220.0, 100)

    constant = fgf.ConstantSED()
    power_law = fgp.PowerLaw()
    cross = fgc.FactorizedCrossSpectrum(constant, power_law)
    cl_spec = cross(nu, nu, ell, alpha=1.0)

    assert isinstance(cl_spec, np.ndarray)
    assert cl_spec.ndim == 3
    assert cl_spec.shape[0] == nu.shape[0]
    assert cl_spec.shape[1] == nu.shape[0]
    assert cl_spec.shape[2] == ell.shape[0]


@pytest.mark.skipif(JAX_SKIP==True, reason="jax not installed")
def test_jax():
    ell = jnp.arange(2, 5000)
    nu = jnp.linspace(90.0, 220.0, 100)

    constant = fgf.ConstantSED()
    power_law = fgp.PowerLaw()
    cross = fgc.FactorizedCrossSpectrum(constant, power_law)
    cl_spec = cross(nu, nu, ell, alpha=1.0)

    assert isinstance(cl_spec, jax.Array)
    assert cl_spec.ndim == 3
    assert cl_spec.shape[0] == nu.shape[0]
    assert cl_spec.shape[1] == nu.shape[0]
    assert cl_spec.shape[2] == ell.shape[0]
