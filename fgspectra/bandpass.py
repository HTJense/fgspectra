from .model import Model


class Bandpass(Model):
    """Bandpasses are a wrapper around SEDs. They change the function signature
    from (nu, **params) into (nu, bp, **params), allowing automatic integration
    over a bandpass without having to change the underlying functionality of
    the SED or the cross-spectrum calculation.

    Usage
    -----

    You can use a bandpass by simply wrapping it around the SED in
    initialization, for example:

    .. code-block :: python

        # Without bandpass
        cross_spectrum = fgc.FactorizedCrossSpectrum(
            fgf.ThermalSZ(),
            fgp.PowerLaw()
        )

        cl = cross_spectrum(nu1, nu2, ell)

        # With bandpass
        cross_spectrum = fgc.FactorizedCrossSpectrum(
            fgb.Bandpass(fgf.ThermalSZ()),
            fgp.PowerLaw()
        )

        cl = cross_spectrum((nu1, bp1), (nu2, bp2), ell)
    """
    def __init__(self, sed, **kwargs):
        super().__init__(**kwargs)
        self.sed = sed
        self.local_parameters = self.sed.local_parameters

    def __call__(self, nus, bps, theta=None, /, *, lib=None, **params):
        is_single = not isinstance(nus, (list, tuple))
        nus = [nus] if is_single else list(nus)
        bps = [bps] if is_single else list(bps)
        if len(nus) != len(bps):
            raise ValueError(f"Number of frequency arrays must equal the "
                             "number of bandpasses!")

        if lib is None:
            pars = [*nus, *bps, *params.values()]
            if theta is not None:
                pars.extend(theta if isinstance(theta, (list, tuple)) else [theta])
            lib = self.backend(*pars)

        params = self.get_params(lib, theta, params)
        params = self._expand_params(lib, params, self.n_axes)
        out = [self.eval(lib.asarray(nu), lib.asarray(bp), lib=lib, **params)
               for nu, bp in zip(nus, bps)]

        if is_single:
            return out[0]

        return lib.stack(out, axis=-1)

    def eval(self, nu, bp, /, *, lib, **params):
        f_nu = self.sed(nu, lib=lib, **params)

        return lib.trapezoid(f_nu * bp, nu) / lib.trapezoid(bp, nu)

    @property
    def n_params(self):
        return self.sed.n_params
