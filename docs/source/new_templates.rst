Adding new Templates
====================

New templates and SEDs can be added quickly and easily. For example, the
code needed to recreate the power law in either power spectrum space is:

.. code-block :: python

    # fgspectra/power.py

    class PowerLaw(PowerModel):
        local_parameters = ["amp", "alpha"]
        template_defaults = {"amp": 1.0, "alpha_0": 0.0}

        def eval(self, ell, /, *, lib, amp, alpha):
            return amp * (ell / self.ell_0) ** (alpha - self.alpha_0)


Let us go through this implementation step-by-step:

First, notice that we are overloading the :code:`PowerModel` type, which is the
base class for power spectrum-type models. Each power spectrum model should
implement its own `eval` function, which should return the :math:`C_\ell`
function. The standard signature for these functions is

.. code-block :: python

    def eval(self, ell: Array[int], /, *, lib: [numpy | jax], **params) -> Array[float]:
        ...

Here, the first (position-only) argument is the multipole number :code:`ell`.
By convention, the multipole number (and for SEDs, the frequency :code:`nu`)
must be the first, position-only argument. Then, we get all keyword-only
arguments, which includes the backend library :code:`lib`. Lib is a stand-in for
either :code:`numpy` or :code:`jax`, and to maintain compatibility with both
codes you **must** use this library at all times, and only call functions that
are implemented in both libraries. After this, the remaining parameters are the
keyword-only parameters referring to the parameterization of the spectrum that
we are interested in. For the power law implemented here, we are seeking to
implement the template
:math:`C_\ell(A, \alpha) = A (\ell / \ell_0)^(\alpha - \alpha_0)`, where
:math:`A` is some amplitude, and :math:`\alpha` is the power law index of the
spectrum. These two parameters are parameters of every spectrum, so they are
included in the function signature. Then we have two normalization parameters,
:math:`\ell_0` and :math:`\alpha_0`, which are the multipole scale of the pivot
point and a pivot value for the spectral index. These two are not parameters
in the same sense: they are values we are most likely going to define at the
initialization of the template, and keep constant throughout an analysis, but
it might be that these numbers are different for different templates (for
example, most templates have their normalization point defined at
:math:`\ell_0 = 3000`, but dust templates often use :math:`\ell_0 = 500`). By
default, :code:`fgspectra` defines three global default normalization constants,
:code:`ell_0 = 3000`, :code:`nu_0 = 150.0`, and :code:`T_CMB = 2.72548`. Further
normalization constants, where desired or where a diffferent default value
is preferred, should be included in the :code:`template_defaults` dictionary
defined at the top of the class definition. These normalization constants are
not passed to the class via the function parameters, but are instead stored
as class-level attributes, and are thus (as seen in the example) accessed
through the :code:`self` object.

It is important that all parameters given in the function level argument list
are also included in the :code:`local_parameters` keyword. This keyword is
needed to identify the template parameters. The order of the parameters is not
important in either location (although the order in the :code:`local_parameters`
list is used as the order in vector-like function calls), so long as each
parameter used in the :code:`eval` function is included in the :code:`local_parameters`
function.


Parameters, Defaults, and Constants
===================================

:code:`fgspectra` differentiates between three types of parameter values given
to templates: mandatory parameters, defaults, and constants. The reason for
this differentiation is due to the way they are handled in the two different
ways in which :code:`fgspectra` objects can be evaluates, namely with keyword
arguments or with a vector of values.

In the example given above, we see all three parameters: :code:`alpha_0` is a
constant of the template, :code:`amp` is a parameter with a default value,
and :code:`alpha` is a mandatory parameter. The distinction in their use cases
is explained below here.


Mandatory parameters
--------------------

The simplest form of parameters are *mandatory* parameters. These parameters
are given in the :code:`local_parameters` list and are passed on to the
:code:`eval` function. When calling a function with keyword arguments or with
a vectorized function call, these parameters must **always** be included.


Default values
--------------

Sometimes, a parameter can be optional, maybe there is some value that you want
to use as the default, but leave the option for the end-user to override it.
In this case, you want to provide it as a *default* value. Default values for
parameters are defined in the same manner as mandatory parameters, but you
can add a default value in the `template_defaults` tag. We see in the example
above that the amplitude parameter :code:`amp` is set to one by default, so if
the user were to call `power_law(ell, alpha=1.0)`, they would get a spectrum
normalized to unity at the pivot scale.

The user can provide default values themselves as well, by passing on the value
at initialization. For example, the user might want to plot a bunch of spectra
normalized at the same value. In this case, the user can pass a default value
at initialization via:

.. code-block :: python

    power_law = fgp.PowerLaw(amp=3.6)
    cls = power_law(ell, alpha=alpha_array)

This same way, the user is also allowed to choose their own default values, for
example to plot a variety of amplitudes, all with the same spectral index:

.. code-block :: python

    power_law = fgp.PowerLaw(alpha=1.0)
    cls = power_law(ell, amp=amp_array)

This way, the end-user is allowed to define their own default values, and these
values take precedence over the template-level definitions.

However, default values are unused in vectorized calls. This is a necessity, as
the number of parameters provided needs to be known beforehand, and therefore
parameters with default values provided are still expected in these calls.

.. code-block :: python

    power_law = fgp.PowerLaw(alpha=1.0)
    # We still need to provide alpha to the power law.
    theta = np.array([ parameters[p] for p in power_law.parameters ]).T
    cls = power_law(ell, theta)


Constant values
---------------

Constant values are values which are only given as a value upon initialization,
and can never be overridden at a later stage. To include a constant value,
it should **only** be included in the :code:`template_defaults` block.
Thereafter, it is stored as an object-level attribute. In the example above,
we define the constant :code:`alpha_0` as a reference spectral index, with its
standard value :code:`alpha_0 = 0`. In the evaluation function, it is
accessed via :code:`self.alpha_0`, rather than being included as a function
argument. This means that :code:`alpha_0` may **never** be passed on in the
function arguments of a power law (which is intended! We are telling the
user of a power law that this is a constant that they should not be varying!).
It is therefore also never included in vectorized calls.

.. code-block :: python

    power_law = fgp.PowerLaw()
    # Not allowed!
    cls = power_law(ell, alpha=alpha_array, alpha_0=1.0)

The end-user is able to define their own constant values. This is done by passing
a constant value dictionary upon initialization of the template, or by overriding
the value of a pre-existing constant.

.. code-block :: python

    # These two are identical.
    power_law = fgp.PowerLaw(alpha_0=1.0)
    power_law = fgp.PowerLaw(constants={"alpha_0": 1.0})
    cls = power_law(ell, alpha=alpha_array)

This also means that the end user is able to exclude parameters from vectorized
calls by defining them as constants upon initialization - this will make the
code keep the constant value fixed and remove it from the required parameters
list.

.. code-block :: python

    power_law = fgp.PowerLaw(constants={"alpha": 1.0})
    # This no longer included alpha in the parameters!
    theta = np.array([ parameters[p] for p in power_law.parameters ]).T
    cls = power_law(ell, theta)
