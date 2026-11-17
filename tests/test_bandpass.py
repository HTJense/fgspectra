import pytest

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_SKIP = False
except Exception as e:
    JAX_SKIP = True

import fgspectra.bandpass as fgb
import fgspectra.frequency as fgf


def test_numpy():
    nu = np.linspace(70.0, 250.0, 100)
    # Gaussian bandpass
    bp = np.ones_like(nu)
    bp = bp / np.trapezoid(bp, nu)

    constant = fgf.ConstantSED()
    fg_bp = fgb.Bandpass(constant)

    f_bp = fg_bp(nu, bp)

    assert isinstance(f_bp, float)

    mu_bp = [90.0, 150.0, 220.0]
    nus = [nu for _ in mu_bp]
    bps = [np.exp(-0.5 * np.power((nu - mu) / 10.0, 2.0)) for nu, mu in zip(nus, mu_bp)]

    f_bp = fg_bp(nus, bps)
    assert isinstance(f_bp, np.ndarray)


@pytest.mark.skipif(JAX_SKIP == True, reason="jax not installed")
def test_jax():
    nu = jnp.linspace(70.0, 250.0, 100)
    # Gaussian bandpass
    bp = jnp.exp(-0.5 * np.power((nu - 150.0) / 10.0, 2.0))
    bp = bp / jnp.trapezoid(bp, nu)

    constant = fgf.ConstantSED()
    fg_bp = fgb.Bandpass(constant)

    f_nu = constant(nu)
    f_bp = fg_bp(nu, bp)

    assert isinstance(f_bp, jax.Array)

    mu_bp = [90.0, 150.0, 220.0]
    nus = [nu for _ in mu_bp]
    bps = [
        jnp.exp(-0.5 * jnp.power((nu - mu) / 10.0, 2.0)) for nu, mu in zip(nus, mu_bp)
    ]

    f_bp = fg_bp(nus, bps)
    assert isinstance(f_bp, jax.Array)
