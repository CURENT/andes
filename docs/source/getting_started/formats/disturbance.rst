.. _disturbance:

Disturbances
------------

.. _disturbance_devices:

Disturbance Devices
...................

Predefined disturbances at specified time can be created by adding the
corresponding devices. Three types of predefined disturbances are supported:

1. Three-phase-to-ground fault on buses. See :ref:`Fault` for details.
2. Connectivity status toggling. Disconnecting, connecting, or reconnecting any
   device, including lines, generators and motors can be implemented by
   :ref:`Toggle`.
3. Alteration of values. See :ref:`Alter` for details.

To use these devices, the time of disturbance needs to be known ahead of the
simulation. The simulation program by default checks the network connectivity
status after any disturbance.

Perturbation File
.................

One can implement any custom disturbance using a perturbation file as discussed
in [Milano2010]_. The perturbation file is a Python script with a function named
``pert``. The example for the perturbation file can be found in
``andes/cases/ieee14/pert.py``.

.. autofunction:: andes.cases.ieee14.pert.pert
    :noindex:

.. [Milano2010] F. Milano, “Power System Modelling and Scripting,” in Power
       Modelling and Scripting, F. Milano, Ed. Berlin, Heidelberg: Springer, pp.
       202-204, 2010.

