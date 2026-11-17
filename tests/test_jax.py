import pytest

try:
    import jax
    import jax.numpy as jnp
except Exception as e:
    pytest.skip(
        f"jax not imported: {str(e)}.",
        allow_module_level=True,
    )

import fgspectra.power as fgp
import fgspectra.frequency as fgf
import fgspectra.cross as fgc


def test_vmap():
    ell = jnp.arange(2, 5000)
    nu = jnp.linspace(80.0, 220.0, 100)

    cl = fgp.PowerLaw(ell_0=500)
    sed = fgf.ConstantSED()
    cross = fgc.FactorizedCrossSpectrum(sed=sed, cl=cl)

    cl_spec = cross(nu, nu, ell, alpha=-0.6)

    alphas = jnp.linspace(-0.6, 0.6, 15)

    cl_mat = jax.vmap(lambda alpha: cross(nu, nu, ell, alpha=alpha))(alphas)

    assert cl_spec.shape == cl_mat.shape[1:]
    assert cl_mat.shape[0] == alphas.shape[0]


def test_argtype():
    ell = jnp.arange(2, 5000)
    nu = jnp.linspace(80.0, 220.0, 100)

    cl = fgp.PowerLaw(amp=1.0, ell_0=500)
    sed = fgf.ConstantSED()
    cross = fgc.FactorizedCrossSpectrum(sed=sed, cl=cl)

    params = {"amp": 1.0, "alpha": -0.6}

    cl_alpha = cross(nu, nu, ell, amp=params["amp"], alpha=params["alpha"])

    theta = jnp.array([params[p] for p in cross.parameters])
    cl_theta = cross(nu, nu, ell, theta)

    assert jnp.all(jnp.isclose(cl_alpha, cl_theta))
