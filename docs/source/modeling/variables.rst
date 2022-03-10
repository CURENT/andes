
Variables
=========
DAE Variables, or variables for short, are unknowns to be solved using numerical or analytical methods.
A variable stores values, equation values, and addresses in the DAE array. The base class for variables is
`BaseVar`.
In this subsection, `BaseVar` is used to represent any subclass of `VarBase` list in the table below.

.. currentmodule:: andes.core.var
.. autosummary::
      :recursive:
      :toctree: _generated

      BaseVar
      ExtVar
      State
      Algeb
      ExtState
      ExtAlgeb
      AliasState
      AliasAlgeb

Note that equations associated with state variables are in the form of
:math:`\mathbf{M} \dot{x} = \mathbf{f(x, y)}`,
where :math:`\mathbf{x}` are the differential variables,
:math:`\mathbf{y}` are the algebraic variables,
and :math:`\mathbf{M}` is the mass matrix, and :math:`\mathbf{f}` are the right-hand
side of differential equations.
Equations associated with algebraic variables take the form of
:math:`0 = \mathbf{g}`, where :math:`\mathbf{g}` are the equation right-hand side



`BaseVar` has two types: the differential variable type `State` and the algebraic variable type `Algeb`.
State variables are described by differential equations, whereas algebraic variables are described by
algebraic equations. State variables can only change continuously, while algebraic variables
can be discontinuous.

Based on the model the variable is defined, variables can be internal or external. Most variables are internal
and only appear in equations in the same model.
Some models have "public" variables that can be accessed by other
models. For example, a `Bus` defines `v` for the voltage magnitude.
Each device attached to a particular bus needs to access the value and impose the reactive power injection.
It can be done with `ExtAlgeb` or `ExtState`, which links with an existing variable from a model or a group.

Variable, Equation and Address
------------------------------
Subclasses of `BaseVar` are value providers and equation providers.
Each `BaseVar` has member attributes `v` and `e` for variable values and equation values, respectively.
The initial value of `v` is set by the initialization routine, and the initial value of `e` is set to zero.
In the process of power flow calculation or time domain simulation, `v` is not directly modifiable by models
but rather updated after solving non-linear equations. `e` is updated by the models and summed up before
solving equations.

Each `BaseVar` also stores addresses of this variable, for all devices, in its member attribute `a`. The
addresses are *0-based* indices into the numerical DAE array, `f` or `g`, based on the variable type.

For example, `Bus` has ``self.a = Algeb()`` as the voltage phase angle variable.
For a 5-bus system, ``Bus.a.a`` stores the addresses of the `a` variable for all
the five Bus devices. Conventionally, `Bus.a.a` will be assigned `np.array([0, 1, 2, 3, 4])`.

Value and Equation Strings
--------------------------
The most important feature of the symbolic framework is allowing to define equations using strings.
There are three types of strings for a variable, stored in the following member attributes, respectively:

- `v_str`: equation string for **explicit** initialization in the form of `v = v_str(x, y)`.
- `v_iter`: equation string for **implicit** initialization in the form of `v_iter(x, y) = 0`
- `e_str`: equation string for (full or part of) the differential or algebraic equation.

The difference between `v_str` and `v_iter` should be clearly noted. `v_str` evaluates directly into the
initial value, while all `v_iter` equations are solved numerically using the Newton-Krylov iterative method.

Values Between DAE and Models
-----------------------------
ANDES adopts a decentralized architecture which provides each model a copy of variable values before equation
evaluation. This architecture allows to parallelize the equation evaluation (in theory, or in practice if one
works round the Python GIL). However, this architecture requires a coherent protocol for updating the DAE arrays
and the ``BaseVar`` arrays. More specifically, how the variable and equations values from model ``VarBase``
should be summed up or forcefully set at the DAE arrays needs to be defined.

The protocol is relevant when a model defines subclasses of `BaseVar` that are supposed to be "public".
Other models share this variable with `ExtAlgeb` or `ExtState`.

By default, all `v` and `e` at the same address are summed up.
This is the most common case, such as a Bus connected by multiple devices: power injections from
devices should be summed up.

In addition, `BaseVar` provides two flags, `v_setter` and `e_setter`, for cases when one `VarBase`
needs to overwrite the variable or equation values.

Flags for Value Overwriting
---------------------------
`BaseVar` have special flags for handling value initialization and equation values.
This is only relevant for public or external variables.
The `v_setter` is used to indicate whether a particular `BaseVar` instance sets the initial value.
The `e_setter` flag indicates whether the equation associated with a `BaseVar` sets the equation value.

The `v_setter` flag is checked when collecting data from models to the numerical DAE array. If
`v_setter is False`, variable values of the same address will be added.
If one of the variable or external variable has `v_setter is True`, it will, at the end, set the values in the
DAE array to its value. Only one `BaseVar` of the same address is allowed to have `v_setter == True`.

A `v_setter` Example
------------------------
A Bus is allowed to default the initial voltage magnitude to 1 and the voltage phase angle to 0.
If a PV device is connected to a Bus device, the PV should be allowed to override the voltage initial value
with the voltage set point.

In `Bus.__init__()`, one has ::

    self.v = Algeb(v_str='1')

In `PV.__init__`, one can use ::

    self.v0 = Param()
    self.bus = IdxParam(model='Bus')

    self.v = ExtAlgeb(src='v',
                      model='Bus',
                      indexer=self.bus,
                      v_str='v0',
                      v_setter=True)

where an `ExtAlgeb` is defined to access `Bus.v` using indexer `self.bus`. The `v_str` line sets the
initial value to `v0`. In the variable initialization phase for `PV`, `PV.v.v` is set to `v0`.

During the value collection into `DAE.y` by the `System` class, `PV.v`, as a final `v_setter`, will
overwrite the voltage magnitude for Bus devices with the indices provided in `PV.bus`.
