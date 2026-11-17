from abc import ABC, abstractmethod

from .model import Model


class CrossModel(Model):
    n_axes = 3

    def eval(self, nu1, nu2, ell, /, *, lib, **params):
        return NotImplemented

    def __call__(self, nu1, nu2, ell, theta=None, /, lib=None, **params):
        return super().__call__((nu1, nu2, ell), theta, params, lib)


class FactorizedCrossSpectrum(CrossModel):
    def __init__(self, sed, cl, **kwargs):
        super().__init__(**kwargs)
        self.sed = sed
        self.cl = cl
        self.children = (sed, cl)

    def evaluate(self, nu1, nu2, ell, /, *, lib, params):
        f_nu1 = self.sed.evaluate(nu1, lib=lib, params=params)
        f_nu2 = self.sed.evaluate(nu2, lib=lib, params=params)
        cl = self.cl.evaluate(ell, lib=lib, params=params)

        return cl * f_nu1 * f_nu2
