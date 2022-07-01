
Blocks
======

Background
----------
The block library contains commonly used blocks (such as transfer functions and
nonlinear functions). Variables and equations are pre-defined for blocks to be
used as "lego pieces" for scripting DAE models. The base class for blocks is
:py:class:`andes.core.block.Block`.

The supported blocks include ``Lag``, ``LeadLag``, ``Washout``,
``LeadLagLimit``, ``PIController``. In addition, the base class for piece-wise
nonlinear functions, ``Piecewise`` is provided. ``Piecewise`` is used for
implementing the quadratic saturation function ``MagneticQuadSat`` and
exponential saturation function ``MagneticExpSat``.

All variables in a block must be defined as attributes in the constructor, just
like variable definition in models. The difference is that the variables are
"exported" from a block to the capturing model. All exported variables need to
placed in a dictionary, ``self.vars`` at the end of the block constructor.

Blocks can be nested as advanced usage. See the following API documentation for
more details.

.. autoclass:: andes.core.block.Block
    :members: define
    :noindex:

Transfer Functions
------------------

The following transfer function blocks have been implemented.
They can be imported to build new models.

Linear
``````
.. autoclass:: andes.core.block.Gain
    :members: define
    :noindex:

.. autoclass:: andes.core.block.GainLimiter
    :members: define
    :noindex:

.. autoclass:: andes.core.block.Piecewise
    :members: define
    :noindex:

.. autoclass:: andes.core.block.HVGate
    :noindex:

.. autoclass:: andes.core.block.LVGate
    :noindex:

.. autoclass:: andes.core.block.DeadBand1
    :noindex:

First Order
```````````

.. autoclass:: andes.core.block.Integrator
    :members: define
    :noindex:

.. autoclass:: andes.core.block.IntegratorAntiWindup
    :members: define
    :noindex:

.. autoclass:: andes.core.block.Lag
    :members: define
    :noindex:

.. autoclass:: andes.core.block.LagAntiWindup
    :members: define
    :noindex:

.. autoclass:: andes.core.block.LagFreeze
    :members: define
    :noindex:

.. autoclass:: andes.core.block.LagAWFreeze
    :members: define
    :noindex:

.. autoclass:: andes.core.block.LagRate
    :members: define
    :noindex:

.. autoclass:: andes.core.block.LagAntiWindupRate
    :members: define
    :noindex:

.. autoclass:: andes.core.block.Washout
    :members: define
    :noindex:

.. autoclass:: andes.core.block.WashoutOrLag
    :members: define
    :noindex:

.. autoclass:: andes.core.block.LeadLag
    :members: define
    :noindex:

.. autoclass:: andes.core.block.LeadLagLimit
    :members: define
    :noindex:

Second Order
````````````
.. autoclass:: andes.core.block.Lag2ndOrd
    :members: define
    :noindex:

.. autoclass:: andes.core.block.LeadLag2ndOrd
    :members: define
    :noindex:

PI Controllers
``````````````
.. autoclass:: andes.core.block.PIController
    :members: define
    :noindex:

.. autoclass:: andes.core.block.PIAWHardLimit
    :members: define
    :noindex:

.. autoclass:: andes.core.block.PITrackAW
    :members: define
    :noindex:

.. autoclass:: andes.core.block.PIFreeze
    :members: define
    :noindex:

.. autoclass:: andes.core.block.PITrackAWFreeze
    :members: define
    :noindex:

.. autoclass:: andes.core.block.PIDController
    :members: define
    :noindex:

.. autoclass:: andes.core.block.PIDAWHardLimit
    :members: define
    :noindex:

.. autoclass:: andes.core.block.PIDTrackAW
    :members: define
    :noindex:

Saturation
----------
.. autoclass:: andes.models.exciter.ExcExpSat
    :members: define
    :noindex:


Naming Convention
-----------------

We loosely follow a naming convention when using modeling blocks. An instance of
a modeling block is named with a two-letter acronym, followed by a number or a
meaningful but short variaiable name. The acronym and the name are spelled in
one word without underscore, as the output of the block already contains ``_y``.

For example, two washout filters can be names ``WO1`` and ``WO2``. In another
case, a first-order lag function for voltage sensing can be called ``LGv``, or
even ``LG`` if there is only one Lag instance in the model.

Naming conventions are not strictly enforced. Expressiveness and concision are
encouraged.
