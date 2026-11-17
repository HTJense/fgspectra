import pytest

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_SKIP = False
except Exception as e:
    JAX_SKIP = True

import fgspectra.power as fgp


def test_float():
    ell = 5

    power_law = fgp.PowerLaw(amp=1.0)
    cl = power_law(ell, alpha=1.0)

    assert isinstance(cl, np.ndarray)
    assert cl.shape == (1,)


def test_numpy():
    ell = np.arange(2, 5000)

    power_law = fgp.PowerLaw(amp=1.0)
    cl = power_law(ell, alpha=1.0)

    assert isinstance(cl, np.ndarray)


@pytest.mark.skipif(JAX_SKIP == True, reason="jax not installed")
def test_jax():
    ell = jnp.arange(2, 5000)

    power_law = fgp.PowerLaw(amp=1.0)
    cl = power_law(ell, alpha=1.0)

    assert isinstance(cl, jax.Array)
