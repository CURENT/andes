
Per Unit System
===============

The bases for AC system are

- :math:`S_b^{ac}`: three-phase power in MVA. By default, :math:`S_b^{ac}=100 MVA` (set by ``System.config.mva``).

- :math:`V_b^{ac}`: phase-to-phase voltage in kV.

- :math:`I_b^{ac}`: current base :math:`I_b^{ac} = \frac{S_b^{ac}} {\sqrt{3} V_b^{ac}}`

The bases for DC system are

- :math:`S_b^{dc}`: power in MVA. It is assumed to be the same as :math:`S_b^{ac}`.

- :math:`V_b^{dc}`: voltage in kV.

Some device parameters are given as per unit values under the device base power and voltage (if applicable).
For example, the Line model :py:mod:`andes.models.line.Line` has parameters ``r``, ``x`` and ``b``
as per unit values in the device bases ``Sn``, ``Vn1``, and ``Vn2``.
It is up to the user to check data consistency.
For example, line voltage bases are typically the same as bus nominal values.
If the ``r``, ``x`` and ``b`` are meant to be per unit values under the system base,
each Line device should use an ``Sn`` equal to the system base mva.

Parameters in device base will have a property value in the Model References page.
For example, ``Line.r`` has a property ``z``, which means it is a per unit impedance
in the device base.
To find out all applicable properties, refer to the "Other Parameters" section of
:py:mod:`andes.core.param.NumParam`.

After setting up the system, these parameters will be converted to per units
in the bases of system base MVA and bus nominal voltages.
The parameter values in the system base will be stored to the ``v`` attribute of the ``NumParam``.
The original inputs in the device base will be moved to the ``vin`` attribute.
For example, after setting up the system, ``Line.x.v`` is the line reactances in per unit
under system base.

Values in the ``v`` attribute is what get utilized in computation.
Writing new values directly to ``vin`` will not affect the values in ``v`` afterwards.
To alter parameters after setting up, refer to example notebook 2.
