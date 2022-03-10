
Parameters
==========

Background
-----------

Parameter is a type of building atom for DAE models.
Most parameters are read directly from an input file and passed to equation,
and other parameters can be calculated from existing parameters.

The base class for parameters in ANDES is `BaseParam`, which defines interfaces for adding values and
checking the number of values. `BaseParam` has its values stored in a plain list, the member attribute `v`.
Subclasses such as `NumParam` stores values using a NumPy ndarray.

An overview of supported parameters is given below.

.. currentmodule:: andes.core.param
.. autosummary::
      :recursive:
      :toctree: _generated

      BaseParam
      DataParam
      IdxParam
      NumParam
      ExtParam
      TimerParam
