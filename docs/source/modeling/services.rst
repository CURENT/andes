

Services
========
Services are helper variables outside the DAE variable list. Services are most often used for storing intermediate
constants but can be used for special operations to work around restrictions in the symbolic framework.
Services are value providers, meaning each service has an attribute ``v`` for storing service values. The
base class of services is :py:mod`BaseService`, and the supported services are listed as follows.

.. currentmodule:: andes.core.service
.. autosummary::
      :recursive:
      :toctree: _generated

      BaseService
      OperationService

+------------------+-----------------------------------------------------------------+
|      Class       |                           Description                           |
+==================+=================================================================+
|  ConstService    | Internal service for constant values.                           |
+------------------+-----------------------------------------------------------------+
|  VarService      | Variable service updated at each iteration before equations.    |
+------------------+-----------------------------------------------------------------+
|  ExtService      | External service for retrieving values from value providers.    |
+------------------+-----------------------------------------------------------------+
|  PostInitService | Constant service evaluated after TDS initialization             |
+------------------+-----------------------------------------------------------------+
|  NumReduce       | The service type for reducing linear 2-D arrays into 1-D arrays |
+------------------+-----------------------------------------------------------------+
|  NumRepeat       | The service type for repeating a 1-D array to linear 2-D arrays |
+------------------+-----------------------------------------------------------------+
|  IdxRepeat       | The service type for repeating a 1-D list to linear 2-D list    |
+------------------+-----------------------------------------------------------------+
|  EventFlag       | Service type for flagging changes in inputs as an event         |
+------------------+-----------------------------------------------------------------+
|  VarHold         | Hold input value when a hold signal is active                   |
+------------------+-----------------------------------------------------------------+
|  ExtendedEvent   | Extend an event signal for a given period of time               |
+------------------+-----------------------------------------------------------------+
|  DataSelect      | Select optional str data if provided or use the fallback        |
+------------------+-----------------------------------------------------------------+
|  NumSelect       | Select optional numerical data if provided                      |
+------------------+-----------------------------------------------------------------+
|  DeviceFinder    | Finds or creates devices linked to the given devices            |
+------------------+-----------------------------------------------------------------+
|  BackRef         | Collects idx-es for the backward references                     |
+------------------+-----------------------------------------------------------------+
|  RefFlatten      | Converts BackRef list of lists into a 1-D list                  |
+------------------+-----------------------------------------------------------------+
|  InitChecker     | Checks initial values against typical values                    |
+------------------+-----------------------------------------------------------------+
|  FlagValue       | Flags values that equals the given value                        |
+------------------+-----------------------------------------------------------------+
|  Replace         | Replace values that returns True for the given lambda func      |
+------------------+-----------------------------------------------------------------+


Internal Constants
---------------------------
The most commonly used service is `ConstService`.  It is used to store an array of constants, whose value is
evaluated from a provided symbolic string. They are only evaluated once in the model initialization phase, ahead
of variable initialization. `ConstService` comes handy when one wants to calculate intermediate constants from
parameters.

For example, a turbine governor has a `NumParam` `R` for the
droop. `ConstService` allows to calculate the inverse of the droop, the gain, and use it in equations. The
snippet from a turbine governor's ``__init__()`` may look like ::

    self.R = NumParam()
    self.G = ConstService(v_str='u/R')

where `u` is the online status parameter. The model can thus use `G` in subsequent variable or equation
strings.

.. autoclass:: andes.core.service.ConstService
    :noindex:

.. autoclass:: andes.core.service.VarService
    :noindex:

.. autoclass:: andes.core.service.PostInitService
    :noindex:

External Constants
------------------------
Service constants whose value is retrieved from an external model or group. Using `ExtService` is
similar to using external variables. The values of `ExtService` will be retrieved once during the
initialization phase before `ConstService` evaluation.

For example, a synchronous generator needs to retrieve the `p` and `q` values from static generators
for initialization. `ExtService` is used for this purpose. In the ``__init__()`` of a synchronous generator
model, one can define the following to retrieve `StaticGen.p` as `p0`::

        self.p0 = ExtService(src='p',
                             model='StaticGen',
                             indexer=self.gen,
                             tex_name='P_0')

.. autoclass:: andes.core.service.ExtService
    :noindex:

Shape Manipulators
-------------------------------------------
This section is for advanced model developer.

All generated equations operate on 1-dimensional arrays and can use algebraic calculations only.
In some cases, one model would use `BackRef` to retrieve 2-dimensional indices and will use such indices to
retrieve variable addresses.
The retrieved addresses usually has a different length of the referencing model and cannot be used directly for calculation.
Shape manipulator services can be used in such case.

`NumReduce` is a helper Service type which reduces a linearly stored 2-D ExtParam into 1-D Service.
`NumRepeat` is a helper Service type which repeats a 1-D value into linearly stored 2-D value based on the
shape from a `BackRef`.

.. autoclass:: andes.core.service.BackRef
    :noindex:

.. autoclass:: andes.core.service.NumReduce
    :noindex:

.. autoclass:: andes.core.service.NumRepeat
    :noindex:

.. autoclass:: andes.core.service.IdxRepeat
    :noindex:

.. autoclass:: andes.core.service.RefFlatten
    :noindex:


Value Manipulation
------------------
.. autoclass:: andes.core.service.Replace
    :noindex:

.. autoclass:: andes.core.service.ParamCalc
    :noindex:


Idx and References
-------------------------------------------
.. autoclass:: andes.core.service.DeviceFinder
    :noindex:

.. autoclass:: andes.core.service.BackRef
    :noindex:

.. autoclass:: andes.core.service.RefFlatten
    :noindex:


Events
----------
.. autoclass:: andes.core.service.EventFlag
    :noindex:

.. autoclass:: andes.core.service.ExtendedEvent
    :noindex:

.. autoclass:: andes.core.service.VarHold
    :noindex:


Flags
------------
.. autoclass:: andes.core.service.FlagCondition
    :noindex:

.. autoclass:: andes.core.service.FlagGreaterThan
    :noindex:

.. autoclass:: andes.core.service.FlagLessThan
    :noindex:

.. autoclass:: andes.core.service.FlagValue
    :noindex:



Data Select
-----------
.. autoclass:: andes.core.service.DataSelect
    :noindex:

.. autoclass:: andes.core.service.NumSelect
    :noindex:


Miscellaneous
-------------
.. autoclass:: andes.core.service.InitChecker
    :noindex:

.. autoclass:: andes.core.service.ApplyFunc
    :noindex:

.. autoclass:: andes.core.service.CurrentSign
    :noindex:

.. autoclass:: andes.core.service.RandomService
    :noindex:

.. autoclass:: andes.core.service.SwBlock
    :noindex:
