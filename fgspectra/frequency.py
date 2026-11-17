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
        ...

    def __call__(self, nu, theta=None, /, *, lib=None, **params):
        return super().__call__((nu,), theta, params, lib)


class ConstantSED(SED):
    parameters = []

    def eval(self, nu, /, *, lib, **kwargs):
        return lib.ones_like(nu)


class PowerLawSED(SED):
    local_parameters = ("beta",)
    template_defaults = {"beta_0": 0.0}

    def eval(self, nu, /, *, lib, beta):
        rj = (nu / self.nu_0) ** (beta - self.beta_0)
        return rj * (_rj2cmb(nu, lib, self.T_CMB) / _rj2cmb(self.nu_0, lib, self.T_CMB))


class Synchrotron(PowerLawSED):
    ...


class ModifiedBlackBody(SED):
    local_parameters = ("beta", "T")
    template_defaults = {"beta_0": 0.0}

    def eval(self, nu, /, *, lib, beta, T):
        x = H_OVER_K * nu / T
        x_0 = H_OVER_K * self.nu_0 / T
        res = (nu / self.nu_0) ** (beta - self.beta_0 + 1.) * lib.expm1(x_0) / lib.expm1(x)
        return res * (_rj2cmb(nu, lib, self.T_CMB) / _rj2cmb(self.nu_0, lib, self.T_CMB))


class CIB(ModifiedBlackBody):
    ...


class ThermalSZ(SED):
    @staticmethod
    def f(nu, T_CMB, lib):
        x = H_OVER_K * nu / T_CMB
        return x / lib.tanh(x / 2.0) - 4.0

    def eval(self, nu, /, *, lib):
        return self.f(nu, self.T_CMB, lib) / self.f(self.nu_0, self.T_CMB, lib)


class FreeFree(SED):
    local_parameters = ["EM", "Te"]

    def eval(self, nu, /, *, lib, EM, Te):
        Teff = (Te / 1e3) ** 1.5
        nu_ff = 255.33e9 * Teff
        g_ff = 1. + lib.log(1. + (nu_ff / nu) ** (math.sqrt(3.) / math.pi))
        # TODO: Need to check the units on this.
        return EM * g_ff
