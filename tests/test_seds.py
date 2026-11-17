import pytest

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_SKIP = False
except Exception as e:
    JAX_SKIP = True

import fgspectra.frequency as fgf


def test_float():
    nu = 150.0

    constant = fgf.ConstantSED()
    f_nu = constant(nu)

    assert isinstance(f_nu, np.ndarray)


def test_numpy():
    nu = np.linspace(90.0, 220.0, 100)

    constant = fgf.ConstantSED()
    f_nu = constant(nu)

    assert isinstance(f_nu, np.ndarray)


@pytest.mark.skipif(JAX_SKIP == True, reason="jax not installed")
def test_jax():
    nu = jnp.linspace(90.0, 220.0, 100)

    constant = fgf.ConstantSED()
    f_nu = constant(nu)

    assert isinstance(f_nu, jax.Array)
