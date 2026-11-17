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


def test_ksz_numpy():
    ell = np.arange(2, 5000)
    nu = np.linspace(90.0, 220.0, 25)

    amp = np.linspace(0.5, 1.5, 10)

    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())

    ksz(nu, nu, ell, amp=1.0)
    ksz(nu, nu, ell, amp=amp)


@pytest.mark.skipif(JAX_SKIP == True, reason="jax not installed")
def test_ksz_jax():
    ell = jnp.arange(2, 5000)
    nu = jnp.linspace(90.0, 220.0, 25)

    amp = jnp.linspace(0.5, 1.5, 10)

    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())

    ksz(nu, nu, ell, amp=1.0)
    ksz(nu, nu, ell, amp=amp)

    jax.vmap(lambda amp: ksz(nu, nu, ell, amp=amp))(amp)


def test_tsz_numpy():
    ell = np.arange(2, 5000)
    nu = np.linspace(90.0, 220.0, 25)

    params = {
        "amp": np.linspace(0.5, 1.5, 10),
    }

    tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.PowerLaw(alpha=1.0))

    tsz(nu, nu, ell, amp=1.0)
    tsz(nu, nu, ell, amp=params["amp"])


def test_tsz_jax():
    import jax
    import jax.numpy as jnp

    ell = jnp.arange(2, 5000)
    nu = jnp.linspace(90.0, 220.0, 25)
    n_vec = 10

    params = {
        "amp": jnp.linspace(0.5, 1.5, n_vec),
    }

    tsz = fgc.FactorizedCrossSpectrum(
        fgf.ThermalSZ(), fgp.PowerLaw(constants={"alpha": 1.0})
    )

    # Test float call
    fc = tsz(nu, nu, ell, amp=1.0)
    assert fc.shape == (len(nu), len(nu), len(ell))

    # Test array call
    ac = tsz(nu, nu, ell, amp=params["amp"])
    assert ac.shape == (n_vec, len(nu), len(nu), len(ell))

    # Test vmap
    vc = jax.vmap(lambda amp: tsz(nu, nu, ell, amp=amp))(params["amp"])
    assert vc.shape == (n_vec, len(nu), len(nu), len(ell))

    # Test vector call
    theta = jnp.array([params[p] for p in tsz.parameters]).T
    vc = tsz(nu, nu, ell, theta)
    assert vc.shape == (n_vec, len(nu), len(nu), len(ell))

    params = {
        "amp": 1.0,
    }
    theta = jnp.array([params[p] for p in tsz.parameters])

    # Test gradient call
    g_tsz = jax.jacfwd(tsz, argnums=3)
    gc = g_tsz(nu, nu, ell, theta)
    assert gc.shape == (len(nu), len(nu), len(ell), len(theta))


def test_cibp_numpy():
    ell = np.arange(2, 5000)
    nu = np.linspace(90.0, 220.0, 25)

    params = {"amp": np.linspace(0.5, 1.5, 10), "beta": np.linspace(1.8, 2.2, 10)}

    cibp = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.Poisson(alpha=1.0))

    cibp(nu, nu, ell, amp=1.0, beta=2.0, T=9.6)
    cibp(nu, nu, ell, amp=params["amp"], beta=params["beta"], T=9.6)


def test_cibp_jax():
    import jax
    import jax.numpy as jnp

    ell = jnp.arange(2, 5000)
    nu = jnp.linspace(90.0, 220.0, 25)
    n_vec = 10

    params = {
        "amp": jnp.linspace(0.5, 1.5, n_vec),
        "beta": jnp.linspace(1.8, 2.2, n_vec),
        "T": 9.6 * jnp.ones(n_vec),
        "alpha": 1.0 * jnp.ones(n_vec),
    }

    cibp = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.Poisson(constants={"alpha": 1.0}))

    # Test float call
    fc = cibp(nu, nu, ell, amp=1.0, beta=2.0, T=9.6)
    assert fc.shape == (len(nu), len(nu), len(ell))

    # Test array call
    ac = cibp(nu, nu, ell, amp=params["amp"], beta=params["beta"], T=9.6)
    assert ac.shape == (n_vec, len(nu), len(nu), len(ell))

    # Test vmap
    vc = jax.vmap(lambda amp, beta: cibp(nu, nu, ell, amp=amp, beta=beta, T=9.6))(
        params["amp"], params["beta"]
    )
    assert vc.shape == (n_vec, len(nu), len(nu), len(ell))

    # Test vector call
    theta = jnp.array([params[p] for p in cibp.parameters]).T
    vc = cibp(nu, nu, ell, theta)
    assert vc.shape == (n_vec, len(nu), len(nu), len(ell))

    params = {
        "amp": 1.0,
        "beta": 2.0,
        "T": 9.6,
        "alpha": 1.0,
    }
    theta = jnp.array([params[p] for p in cibp.parameters])

    # Test gradient call
    g_cibp = jax.jacfwd(cibp, argnums=3)
    gc = g_cibp(nu, nu, ell, theta)
    assert gc.shape == (len(nu), len(nu), len(ell), len(theta))


def test_dust_numpy():
    ell = np.arange(2, 5000)
    nu = np.linspace(90.0, 220.0, 25)

    params = {"amp": np.linspace(0.5, 1.5, 10), "beta": np.linspace(1.3, 2.7, 10)}

    dust = fgc.FactorizedCrossSpectrum(
        fgf.ModifiedBlackBody(T=19.6), fgp.DustPowerLaw(alpha=-0.6)
    )

    assert dust.cl.ell_0 == 500

    dust(nu, nu, ell, amp=1.0, beta=1.5)
    dust(nu, nu, ell, amp=params["amp"], beta=params["beta"])


def test_dust_jax():
    import jax
    import jax.numpy as jnp

    ell = jnp.arange(2, 5000)
    nu = jnp.linspace(90.0, 220.0, 25)
    n_vec = 10

    dust = fgc.FactorizedCrossSpectrum(
        fgf.ModifiedBlackBody(constants={"T": 19.6}),
        fgp.DustPowerLaw(constants={"alpha": -0.6}),
    )

    assert dust.cl.ell_0 == 500

    params = {
        "amp": jnp.linspace(0.5, 1.5, 10),
        "beta": jnp.linspace(1.3, 1.7, 10),
    }

    # Test float call
    fc = dust(nu, nu, ell, amp=1.0, beta=2.0)
    assert fc.shape == (len(nu), len(nu), len(ell))

    # Test array call
    ac = dust(nu, nu, ell, amp=params["amp"], beta=params["beta"])
    assert ac.shape == (n_vec, len(nu), len(nu), len(ell))

    # Test vmap
    vc = jax.vmap(lambda amp, beta: dust(nu, nu, ell, amp=amp, beta=beta))(
        params["amp"], params["beta"]
    )
    assert vc.shape == (n_vec, len(nu), len(nu), len(ell))

    # Test vector call
    theta = jnp.array([params[p] for p in dust.parameters]).T
    vc = dust(nu, nu, ell, theta)
    assert vc.shape == (n_vec, len(nu), len(nu), len(ell))

    params = {
        "amp": 1.0,
        "beta": 2.0,
        "T": 19.6,
        "alpha": 1.0,
    }
    theta = jnp.array([params[p] for p in dust.parameters])

    # Test gradient call
    g_dust = jax.jacfwd(dust, argnums=3)
    gc = g_dust(nu, nu, ell, theta)
    assert gc.shape == (len(nu), len(nu), len(ell), len(theta))
