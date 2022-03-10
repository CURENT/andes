
Atomic Types
============
ANDES contains three types of atom classes for building DAE models.
These types are parameter, variable and service.

Value Provider
--------------

Before addressing specific atom classes, the terminology `v-provider`, and `e-provider` are discussed.
A value provider class (or `v-provider` for short) references any class with a member attribute named ``v``,
which should be a list or a 1-dimensional array of values.
For example, all parameter classes are v-providers, since a parameter class should provide
values for that parameter.

.. note::
    In fact, all types of atom classes are v-providers, meaning that an instance of an atom class must contain values.

The values in the `v` attribute of a particular instance are values that will substitute the instance for computation.
If in a model, one has a parameter ::

    self.v0 = NumParam()
    self.b = NumParam()

    # where self.v0.v = np.array([1., 1.05, 1.1]
    #   and self.b.v  = np.array([10., 10., 10.]

Later, this parameter is used in an equation, such as ::

    self.v = ExtAlgeb(model='Bus', src='v',
                      indexer=self.bus,
                      e_str='v0 **2 * b')

While computing `v0 ** 2 * b`, `v0` and `b` will be substituted with the values in `self.v0.v` and `self.b.v`.

Sharing this interface `v` allows interoperability among parameters and variables and services.
In the above example, if one defines `v0` as a `ConstService` instance, such as ::

    self.v0 = ConstService(v_str='1.0')

Calculations will still work without modification.

Equation Provider
-----------------
Similarly, an equation provider class (or `e-provider`) references any class with a member attribute named ``e``,
which should be a 1-dimensional array of values.
The values in the `e` array are the results from the equation and will be summed to the numerical DAE at the addresses
specified by the attribute `a`.

.. note::
    Currently, only variables are `e-provider` types.

If a model has an external variable that links to Bus.v (voltage), such as ::

    self.v = ExtAlgeb(model='Bus', src='v',
                      indexer=self.bus,
                      e_str='v0 **2 * b')

The addresses of the corresponding voltage variables will be retrieved into `self.v.a`,
and the equation evaluation results will be stored in `self.v.e`
