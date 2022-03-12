Group
======
A group is a collection of similar functional models with common variables and parameters.
It is mandatory to enforce the common variables and parameters when develop new models.
The common variables and parameters are typically the interface when connecting different group models.

For example, the Group `RenGen` has variables `Pe` and `Qe`, which are active power output and reactive power output.
Such common variables can be retrieved by other models, such as one in the
Group `RenExciter` for further calculation.

In such a way, the same variable interface is realized so that all model in the same group could carry out similar
function.


.. currentmodule:: andes.models.group
.. autosummary::
      :recursive:
      :toctree: _generated

      GroupBase
