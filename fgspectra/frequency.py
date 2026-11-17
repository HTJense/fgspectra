from abc import ABC, abstractmethod
import math

from scipy import constants

from .model import Model

H_OVER_K = constants.h * 1e9 / constants.k


def _flux2cmb(nu, lib, T_CMB):
    """Converts flux to thermodynamics units"""
    x = H_OVER_K * nu / T_CMB
    g2_min1 = (
        2.0
        * constants.k**3
        * T_CMB**2
        * x**4
        * lib.exp(x)
        / (constants.h * constants.c * lib.expm1(x)) ** 2
    )
    return 1.0 / g2_min1


def _rj2cmb(nu, lib, T_CMB):
    x = H_OVER_K * nu / T_CMB
    return (lib.expm1(x) / x) ** 2 / lib.exp(x)


class SED(Model):
    @abstractmethod
    def eval(self, nu, /, *, lib, **params):
        r"""Evaluate the spectral energy distribution function."""
        ...

    def __call__(self, nu, theta=None, /, *, lib=None, **params):
        return super().__call__((nu,), theta, params, lib)


class ConstantSED(SED):
    r"""Constant value SED."""

    def eval(self, nu, /, *, lib, **kwargs):
        return lib.ones_like(nu)


class PowerLawSED(SED):
    local_parameters = [
        "beta",
    ]
    template_defaults = {"beta_0": 0.0}

    def eval(self, nu, /, *, lib, beta):
        rj = (nu / self.nu_0) ** (beta - self.beta_0)
        return rj * (_rj2cmb(nu, lib, self.T_CMB) / _rj2cmb(self.nu_0, lib, self.T_CMB))


class Synchrotron(PowerLawSED):
    r"""Synchrotron

    Alias of :class:`PowerLawSED`.
    """

    ...


class ModifiedBlackBody(SED):
    local_parameters = ["beta", "temp"]
    template_defaults = {"beta_0": 0.0}

    def eval(self, nu, /, *, lib, beta, temp):
        x = H_OVER_K * nu / temp
        x_0 = H_OVER_K * self.nu_0 / temp
        res = (
            (nu / self.nu_0) ** (beta - self.beta_0 + 1.0)
            * lib.expm1(x_0)
            / lib.expm1(x)
        )
        return res * (
            _rj2cmb(nu, lib, self.T_CMB) / _rj2cmb(self.nu_0, lib, self.T_CMB)
        )


class CIB(ModifiedBlackBody):
    r"""Cosmic Infrared Background

    Alias of :class:`ModifiedBlackBody`.
    """

    ...


class ThermalSZ(SED):
    r"""Thermal Sunyaev-Zel'dovich in K_CMB

    This class implements the

    .. math:: f(\nu) = x \coth(x/2) - 4

    where :math:`x = h \nu / k_B T_CMB`
    """

    @staticmethod
    def f(nu, T_CMB, lib):
        x = H_OVER_K * nu / T_CMB
        return x / lib.tanh(x / 2.0) - 4.0

    def eval(self, nu, /, *, lib):
        """Compute the SED with the given frequency and parameters."""
        return self.f(nu, self.T_CMB, lib) / self.f(self.nu_0, self.T_CMB, lib)


class FreeFree(SED):
    r"""Free-free

    .. math:: f(\nu) = EM * ( 1 + log( 1 + (\nu_{ff} / \nu)^{3/\pi} ) )
    .. math:: \nu_{ff} = 255.33e9 * (Te / 1000)^{3/2}
    """

    local_parameters = ["EM", "Te"]

    def eval(self, nu, /, *, lib, EM, Te):
        """Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            Frequency in the same units as `nu_0`. If array, the shape is
            ``(freq)``.
        EM: float or array
            Emission measure in cm^-6 pc (usually around 300). If array, the shape is ``(...)``.
        Te: float or array
            Electron temperature (typically around 7000). If array, the shape is ``(...)``.
        """
        Teff = (Te / 1e3) ** 1.5
        nu_ff = 255.33e9 * Teff
        g_ff = 1.0 + lib.log(1.0 + (nu_ff / nu) ** (math.sqrt(3.0) / math.pi))
        # TODO: Need to check the units on this.
        return EM * g_ff
