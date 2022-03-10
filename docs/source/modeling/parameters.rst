
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

+---------------+----------------------------------------------------------------------------+
|  Subclasses   |     Description                                                            |
+===============+============================================================================+
|  DataParam    | An alias of `BaseParam`. Can be used for any non-numerical parameters.     |
+---------------+----------------------------------------------------------------------------+
|  NumParam     | The numerical parameter type. Used for all parameters in equations         |
+---------------+----------------------------------------------------------------------------+
|  IdxParam     | The parameter type for storing `idx` into other models                     |
+---------------+----------------------------------------------------------------------------+
|  ExtParam     | Externally defined parameter                                               |
+---------------+----------------------------------------------------------------------------+
|  TimerParam   | Parameter for storing the action time of events                            |
+---------------+----------------------------------------------------------------------------+

Data Parameters
---------------
.. autoclass:: andes.core.param.BaseParam
    :noindex:

.. autoclass:: andes.core.param.DataParam
    :noindex:

.. autoclass:: andes.core.param.IdxParam
    :noindex:

Numeric Parameters
------------------
.. autoclass:: andes.core.param.NumParam
    :noindex:

External Parameters
-------------------
.. autoclass:: andes.core.param.ExtParam
    :noindex:

Timer Parameter
---------------
.. autoclass:: andes.core.param.TimerParam
    :noindex:
