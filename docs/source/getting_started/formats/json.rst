
.. _`input-json`:

ANDES JSON
----------

Overview
........

JSON is a portable format for storing data. It has been used in several other
power system tools, including PowerModels_, Pandapower_, NREL-SIIP_, and GridCal_.
It must be noted that JSON files from these tools are not interoperable because
JSON only defines the data structure, not the meaning of data.

.. _PowerModels: https://lanl-ansi.github.io/PowerModels.jl/stable/
.. _Pandapower: https://www.pandapower.org/
.. _NREL-SIIP: https://github.com/nrel-siip
.. _GridCal: https://gridcal.readthedocs.io

Compared with the `xlsx` file which is a zipped package, the ANDES JSON file is
much faster to parse. We recommend that you use JSON in the following scenarios:

- Your test case is stable and require no manual editing, or
- You will read/write a large number of cases

To convert ``kundur_full.xlsx`` to the ANDES JSON format, do

.. code:: bash

    andes run kundur_full.xlsx -c json

The output file will be named ``kundur_full.json``.

Data storage
............

The ANDES JSON format uses one large dictionary for all devices in the system.
The keys of the dictionary are the model names, and the values are lists of
dictionaries. In each dictionary, the keys are the parameter names and the
values are the parameter values.

The following shows the structure of a JSON file:

.. code:: javascript

    {
    "Toggler": [
        {
        "idx": 1,
        "u": 1.0,
        "name": "Toggler_1",
        "model": "Line",
        "dev": "Line_8",
        "t": 2.0
        }  //      <- Toggler_1 ends
    ],     //      <- Toggler model ends
    "Bus": [
        {
        "idx": 1,
        "u": 1.0,
        "name": 1,
        "Vn": 20.0,
        "vmax": 1.1,
        "vmin": 0.9,
        ...  //    <- other parameters are omitted
        },
        {
        "idx": 2,
        "u": 1.0,
        "name": 2,
        "Vn": 20.0,
        "vmax": 1.1,
        "vmin": 0.9,
        ...  //    <- other parameters are omitted
        },
        ...  //    <- other buses

    ],   //        <- Bus model ends
    ...  //        <- other models
    }    //        <- whole system ends

There are thirdparty tools for editing JSON files, but we still recommend to
convert files to ``xlsx`` for editing. The conversion can be readily done with

.. code:: bash

    andes run kundur_full.json -c xlsx


