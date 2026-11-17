from abc import ABC, abstractmethod
import os
import importlib.resources as ilr

import numpy as np

from .model import Model

def _get_power_file(model):
    """File path for the named model"""
    file_path = ilr.files("fgspectra").joinpath("data", f"cl_{model}.dat")
    if file_path.is_file():
        return str(file_path)
    raise ValueError("No template for model " + model)


class PowerModel(Model):
    @abstractmethod
    def eval(self, ell, /, *, lib, **params):
        ...

    def __call__(self, ell, theta=None, /, *, lib=None, **params):
        return super().__call__((ell,), theta, params, lib)


class PowerSpectrumFromFile(PowerModel):
    """Power spectrum loaded from file(s)

    Parameters
    ----------
    filenames: string or array of strings
        File(s) to load.
    """

    local_parameters = ["amp"]

    def __init__(self, filenames, **kwargs):
        super().__init__(**kwargs)

        filenames = np.array(filenames)
        self.templates = np.empty(filenames.shape + (0,))

        for i, filename in np.ndenumerate(filenames):
            ell, spec = np.genfromtxt(filename, unpack=True)
            ell = ell.astype(int)
            # Make sure that the new spectrum fits self._cl
            n_missing_ells = ell.max() + 1 - self.templates.shape[-1]
            if n_missing_ells > 0:
                pad_width = [(0, 0)] * self.templates.ndim
                pad_width[-1] = (0, n_missing_ells)
                self.templates = np.pad(
                    self.templates, pad_width, mode="constant",
                    constant_values=0
                )

            self.templates[i + (ell,)] = spec

        self.cl_0 = self.templates[..., self.ell_0]

    def eval(self, ell, /, *, lib, amp):
        return amp * lib.asarray(self.templates)[..., ell] / self.cl_0


class PowerLawExtendedTemplate(PowerSpectrumFromFile):
    """Power spectrum loaded from a file, extended to a power law above the
    pivot point :math:`\\ell_0`.
    """
    local_parameters = ("amp", "alpha")
    template_defaults = {"alpha_0": 0.0}

    def eval(self, ell, /, *, lib, amp, alpha):
        return lib.where(
           ell < self.ell_0,
           amp * lib.asarray(self.templates)[..., ell] / self.cl_0,
           amp * (ell / self.ell_0) ** (alpha - self.alpha_0)
        )


class PowerLawRescaledTemplate(PowerSpectrumFromFile):
    """Power spectrum loaded from a file, rescaled by a power law.
    """
    local_parameters = ("amp", "alpha")
    template_defaults = {"alpha_0": 0.0}

    def eval(self, ell, /, *, lib, amp, alpha):
        cl = amp * lib.asarray(self.templates)[..., ell] / self.cl_0
        return cl * (ell / self.ell_0) ** (alpha - self.alpha_0)


class tSZ_150_bat(PowerSpectrumFromFile):
    """Powerspectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""
    def __init__(self, **kwargs):
        super().__init__(_get_power_file("tsz_150_bat"), **kwargs)


class kSZ_bat_full(PowerSpectrumFromFile):
    """Powerspectrum for Kinematic Sunyaev-Zel'dovich (Dunkley et al. 2013).
    Contains both contributions from late-time and reionization kSZ."""
    def __init__(self, **kwargs):
        super().__init__(_get_power_file("ksz_bat_full"), **kwargs)


class kSZ_bat(PowerSpectrumFromFile):
    """Powerspectrum for Kinematic Sunyaev-Zel'dovich (Dunkley et al. 2013)."""
    def __init__(self, **kwargs):
        super().__init__(_get_power_file("ksz_bat"), **kwargs)


class PowerLaw(PowerModel):
    r"""Power law

    .. math :: C_\ell = A (\ell / \ell_0)^{\alpha - \alpha_0}
    """

    local_parameters = ("amp", "alpha")
    template_defaults = {"amp": 1.0, "alpha_0": 0.0}

    def _x(self, ell):
        return ell

    def eval(self, ell, /, *, lib, amp, alpha):
        return amp * (self._x(ell) / self._x(self.ell_0)) ** (alpha - self.alpha_0)


class DustPowerLaw(PowerLaw):
    r"""Power law with pivot scale :math:`\ell_0 = 500`, defined for convenience.
    """
    template_defaults = {"alpha_0": 0.0, "ell_0": 500}


class Poisson(PowerLaw):
    r"""Power law for poisson-like components.

    .. math :: C_\ell = A \left( \frac{\ell (\ell+1)}{\ell_0 (\ell_0 + 1)} \right)^{\alpha - \alpha_0}
    """
    def _x(self, ell):
        return ell * (ell + 1.)
