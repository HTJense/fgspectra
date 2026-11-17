from abc import ABC, abstractmethod
from functools import cached_property
import inspect

import numpy as np
import yaml


NAME_SEPARATOR = "_"

# Global, package-wide defaults.
global_defaults = {
    "ell_0": 3000,
    "nu_0": 150.,
    "T_CMB": 2.72548,
}


class Model(ABC):
    _default_backend = np
    local_parameters: list = []
    template_defaults: dict = {}
    n_axes: int = 1

    def __init__(self, *, name=None, renames=None, constants=None, **defaults):
        self.name = name
        self.renames = dict(renames or {})
        self.constants = dict(constants or {})
        self.children = []

        # Check that all given parameters are known to this code.
        unknown = (set(self.renames) | set(self.constants)) - set(self.local_parameters)
        if len(unknown) > 0:
            raise ValueError(
                f"{type(self).__name__}: unknown parameters: {unknown}. "
                f"Recognized parameters: {self.local_parameters}"
            )

        # Check if there is overlap between some provided parameters.
        if len(overlap := set(defaults) & set(self.constants)
                           & set(self.local_parameters)) > 0:
            raise ValueError(
                f"{type(self).__name__}: parameters: {overlap} given as both "
                "default values and fixed values. Choose one!"
            )

        self.defaults = {
            **global_defaults,
            **self.template_defaults,
            **defaults
        }

        self.param_defaults = {
            p: self.defaults.pop(p)
            for p in list(self.defaults)
            if p in self.local_parameters
        }

        # Remove constants from defaults.
        for p in self.constants:
            self.param_defaults.pop(p, None)

        for param in self.defaults:
            if hasattr(type(self), param):
                raise ValueError(f"Parameter {param} is a reserved class-level"
                                 " attribute. Please use a different name!")
            setattr(self, param, self.defaults[param])

    def backend(self, *inputs):
        """Checks which backend should be used, supports both Numpy and JAX,
        but could theoretically handle many array-like libraries.

        Defaults to using numpy.
        """
        for x in inputs:
            if type(x).__module__.startswith("jax"):
                import jax.numpy as jnp
                return jnp

        for x in inputs:
            if hasattr(x, "__array_namespace__"):
                return x.__array_namespace__()

        return self._default_backend

    def get_params(self, lib, theta, params):
        if theta is not None:
            if params:
                raise TypeError("Both theta and kwargs provided: choose one!")
            theta = lib.asarray(theta)
            if theta.shape[-1] != len(self.parameters):
                raise ValueError(f"Expected {len(self.parameters)} "
                                 f"parameters ({self.parameters}), but "
                                 f"{theta.shape} were given.")
            return {
                par: theta[..., i]
                for i, par in enumerate(self.parameters)
            }

        unknown = set(params) - set(self.parameters)
        if len(unknown) > 0:
            raise TypeError(f"Unknown parameters {unknown}.")
        defaults = self.global_parameter_defaults
        missing = [
            p
            for p in self.parameters
            if p not in params and p not in defaults
        ]
        if len(missing) > 0:
            raise TypeError(f"Missing parameters {missing}")
        return {
            p: (params[p] if p in params else defaults[p])
            for p in self.parameters
        }

    @staticmethod
    def expand_parameters(lib, params, n):
        """Turns a parameter array into a dictionary with an extra `n`
        new singleton axes added at the end.
        """
        out = {}
        for p in params:
            v = lib.asarray(params[p])
            out[p] = lib.reshape(v, tuple(v.shape) + (1,) * n)
        return out

    @staticmethod
    def expand_axes(lib, *axes):
        """Helper function to expand a set of evaluation axes into a grid of
        evaluation points.
        Given a list of `axes`, e.g. `axes = (nu, ell)`, where `nu.shape = (n,)`
        and `ell.shape = (m,)`, returns a list of the same axes but with
        additional singleton axes, i.e. the result `nu.shape = (n,1)` and
        `ell.shape = (1,m)`.
        """
        n = len(axes)
        axes = [lib.asarray(ax) for ax in axes]
        return [
            lib.reshape(ax, tuple(ax.shape) + (1,) * (n - i - 1))
            for i, ax in enumerate(axes)
        ]

    def global_name(self, local):
        """Rename a parameter that this model can use to a name that the
           top-level model should understand."""
        if local in self.renames:
            return self.renames[local]
        elif self.name:
            return local + NAME_SEPARATOR + self.name
        return local

    def local_name(self, params):
        """Turn a dictionary of global parameters into a dictionary of
        local parameters (with values).

        For example, suppose we have a tSZxCIB cross-spectrum that is given
        a set of parameters, say `amp, beta_cib, alpha_cib, alpha_tsz`,
        where `amp` and `alpha_tsz` need to go to the tSZ template,
        `alpha_cib` to the CIB template, and `beta_cib` to the CIB SED. Then,
        if those four parameters are all given to each of the different
        components, this function will sort out which parameters each
        component needs and which ones it does not need.
        """
        out = dict(self.constants)
        for p in self.local_parameters:
            if p not in self.constants:
                out[p] = params[self.global_name(p)]
        return out

    def __call__(self, axes, theta, params, lib):
        if lib is None:
            pars = list(axes) + list(params.values())
            if theta is not None:
                pars.extend(
                    theta if isinstance(theta, (list, tuple)) else [theta]
                )
            lib = self.backend(*pars)

        params = self.get_params(lib, theta, params)
        params = self.expand_parameters(lib, params, self.n_axes)
        axes = self.expand_axes(lib, *axes)
        return self.evaluate(*axes, params=params, lib=lib)

    def evaluate(self, *args, params, lib):
        return self.eval(*args, lib=lib, **self.local_name(params))

    @abstractmethod
    def eval(self, *args, lib, **params):
        ...

    @cached_property
    def global_parameter_defaults(self):
        defaults = {}

        def add_default(p, v):
            if p in defaults and not np.array_equal(defaults[p], v):
                raise ValueError("Conflicting defaults for shared parameter.")

            defaults[p] = v

        for c in self.children:
            for p in (c_defaults := c.global_parameter_defaults):
                add_default(p, c_defaults[p])

        for p in self.param_defaults:
            add_default(self.global_name(p), self.param_defaults[p])
        for p in self.constants:
            add_default(self.global_name(p), self.constants[p])

        return defaults

    def records(self):
        recs = []
        for p in self.local_parameters:
            if p not in self.constants:
                recs.append((self.global_name(p), p in self.renames, self))
        for c in self.children:
            recs.extend(c.records())
        return recs

    @cached_property
    def free_parameters(self):
        owners = {}
        for p, explicit, owner in self.records():
            prev = owners.get(p)
            if prev is None:
                owners[p] = (explicit, owner)
            elif prev[1] is not owner and not (prev[0] and explicit):
                raise ValueError(
                    f"Parameter {p} is used by multiple components. You can "
                    f"differentiate them through the `name=...` parameter, "
                    f"or mark them as the same parameter with `renames=...`."
                )
        return list(owners)

    @property
    def parameters(self):
        return self.free_parameters

    @property
    def n_params(self):
        return len(self.free_parameters)

    def __repr__(self):
        return (f"{type(self).__name__}(name={self.name!r}, "
                f"parameters={self.parameters}, defaults={self.defaults})")
