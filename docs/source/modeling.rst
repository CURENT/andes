.. _modeling:

***************
Device Modeling
***************


Base ``Model`` and ``ModelData`` Class
======================================
The `ModelData` class provides structure and methods for storing
power system data incrementally.

The `Model` class provides functions needed for defining
variables and equations.

OrderedDict of instances
-------------------------

Variables:

Variables has the following attributes in common:
 - *a* for variable address
 - *v* for variable value
 - *e* for the corresponding equation value
 - *e_symbolic* for the string/symbolic representation of the equation
 - *e_numeric* for the callable to update equation value
 - *e_lambdify* for the generated callable to update equation value

ExtVar:

External variables has the additional method:
 - *link_external()* for linking to external variable

The following variable containers exist:
 - *states* for differential variables
 - *algebs* for algebraic variables
 - *calcs* for calculated variables
 - *vars_ext* for external variables

Parameters:

ParamBase hold the following attributes:
 - *property* for a dictionary of properties for data requirements
 - *v* for a list/array of values from input
 - *get_name()* returns a list only containing its name

NumParam holds the following additional attributes:
 - *pu_coeff* for coefficients for per-unit conversion
 - *vin* for a copy of the input variables
 - *params* for internal parameters
 - *params_ext* for external parameters

ExtParam holds the additional methods:
 - *link_external* for linking external parameter data

Service Constants:
 - *services* for service constants

Limiters:
Limiters are used to add limits to algebraic or state variables. 
Limiters need be provided with a variable and its limits. 
 - *limiters* for limiters

Blocks:

Blocks are collections of variables and the corresponding equations.
Blocks can be instantiated as model attributes. The instantiation of blocks
will add the corresponding variables and equations to the parent class.
An example block is the PIController. 
 - *blocks* for general blocks