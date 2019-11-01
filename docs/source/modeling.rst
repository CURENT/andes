.. _modeling:

***************
Device Modeling
***************


Base ``Model`` Class
====================

OrderedDict of instances
-------------------------

Variables:

Variables has the following attributes in common:
 - *a* for variable address
 - *v* for variable address
 - *e* for variable equation value
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
 - *limiters* for limiter blocks

Blocks:
 - *blocks* for general blocks