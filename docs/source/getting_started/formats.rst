=============
Input formats
=============

ANDES currently supports the following input formats:

- ``.xlsx``: Excel spreadsheet file with ANDES data
- ``.json``: JSON plain-text file with ANDES semantics
- ``.raw`` and ``.dyr``: PSS/E raw and dyr files
- ``.m``: MATPOWER case file

ANDES xlsx
----------

The ANDES xlsx format uses Microsoft Excel for conveniently viewing and editing
model parameters. If you don't have access to Excel, you can use the free and
open-source `LibreOffice <https://www.libreoffice.org>`_ or the non-paid `WPS
Office <https://www.wps.com/>`_.

Format definition
.................

The ANDES xlsx format contains multiple workbooks (aka sheet) shown as tabs at
the bottom. The name of a workbook is a *model* name, and each workbook contains
the parameters of all *devices* that are *instances* of the model.

In each sheet, the first row contains the names of parameters of the model.
Starting from the second row, each row corresponds to a *device instance* with
the parameters in the corresponding columns. An example of the ``Bus`` sheet
is shown in the following screenshot.

.. image:: tutorial/xlsx-bus.png
   :width: 600
   :alt: Example workbook for Bus

Common parameters
.................

A few columns are used across all models. That includes ``uid``, ``idx``,
``name`` and ``u``:

- ``uid`` is an unique index that is generated and used *internally*. This
  column can be left empty when the sheet is being created manually. Exporting
  systems to ``xlsx`` with ``--convert`` (see :ref:`format-converter`) will have
  the ``uid`` overwritten.
- ``idx`` is the *unique index to identify a device* of the model. An unique
  ``idx`` should be provided explicitly for each instance for best consistency.
  Accepted types for ``idx`` include numbers and strings without spaces.

.. warning ::

    ANDES will check the uniqueness of ``idx`` and assign new ones when
    duplication is detected. Duplicate ``idx`` indicates data inconsistency and
    will likely cause simulations to fail.

- ``u`` is the connectivity status of the instance. Accepted values are 0 for
  disconnected (turned off) and 1 for connected (turned on). Disconnected
  devices will still have the variables assigned in ANDES but will not interact
  with the simulation. Unexpected behaviors may occur if numerical values other
  than 0 and 1 are assigned, as ``u`` is often used as a multiplier in equations.
- ``name`` is the name for the device instance. It is used for display purposes
  and can be left empty.

Connecting devices
..................
Most importantly, ``idx`` is the unique index to *connect* a device. In a system, a
PQ (constant PQ load) device needs to connect to a Bus device to inject power.
That is, the PQ device needs to indicate the Bus device to which it is
connected. Such connection is done in the ``PQ`` sheet by setting the ``bus``
parameter to the ``idx`` of the connected bus.

.. image:: tutorial/xlsx-pq.png
   :width: 600
   :alt: Example workbook for PQ

In the above example PQ workbook, there are two PQ instances called ``PQ_0`` and
``PQ_1`` (referred to by ``idx``). They are connected to buses ``7`` and ``8``.
Therefore, on the ``Bus`` sheet, two rows need to exist with ``idx`` being ``7``
and ``8``.

Creating cases
..............

It is often easier to modify from existing cases than creating from scratch. We
recommend that you get familiar with the cases available with ANDES, see
:ref:`test-cases`.

Adding devices
..............

Adding devices to an existing workbook is straightforward. Navigate to the sheet
corresponding to the model and add a new line below the existing lines.

Almost all models have so-called mandatory parameters. They are essential to
describe a complete and consistent test case. For example, the ``PQ`` model has
the ``bus`` parameter as mandatory to indicate the connected bus. To look up
mandatory parameters, see :ref:`modelref` or use ``andes doc MODEL_NAME``.
Check for "mandatory" in the last column called "Properties". This column also
contains other data consistency requirements discussed in the following.

Non-mandatory parameters are optional, meaning that if not provided, ANDES will
use the default parameters. The default values can also be found in
:ref:`modelref`. This does not mean that such parameters should always be left
blank. For example, the ``p0`` (active power load) of ``PQ`` is optional, but
likely one wants to set it to a non-zero value.

There are consistency requirements for parameters, such as ``non_zero``,
``non_negative`` or ``non_positive``. If unmet, the default values will be used.
See the class reference in :py:mod:`andes.core.param.NumParam`.

Autofill data
.............
When you finished adding devices but left some optional parameters empty, you
can use ANDES to autofill them. This is useful when you want to populate a large
number of devices with the same parameters that can be modified later.

The autofill is done through the data converter, namely, ``--convert`` or
``-c``. ANDES will read in the Excel file, fill the optional parameters with
default values, fix the inconsistent values, and then export the data back to
Excel.

.. warning::

    Please backup the spreadsheet if it contains customized edits. Inconsistent
    data will be replaced during the conversion. Formatting in the spreadsheet
    will be lost. Unrecognized sheets will also be discarded.

To autofill ``kundur_full.xlsx``, do

.. code:: bash

    andes run kundur_full.xlsx -c

You will be prompted to confirm the overwrite.

Since this autofill feature utilizes the converter, the autofilled data can be
exported to other formats, such as ``.json``. To do so, use ``-c json``.

Adding workbooks
................

If one wants to add workbooks for models that does not exist in an xlsx file,
one can use ``--add-book ADD_BOOK`` (or ``-b ADD_BOOK``), where ``ADD_BOOK`` can
be a single model name or comma-separated model names (*without space*). For
example,

.. code:: bash

    andes run kundur_full.xlsx -b Fault

will add an empty ``Fault`` sheet to ``kundur_full.xlsx``.

.. Warning::

    With ``--add-book``, the xlsx file will be overwritten with the same
    parameter corrections as in the autofill. Please make backups as needed.

Format conversion and workbook addition can be performed together. To convert a
PSS/E raw file and a dyr file into an xlsx file and add a workbook for ``Fault``, do

.. code:: bash

    andes run kundur.raw -addfile kundur_full.dyr -c -b Fault

The output will have the same name as the raw file.

Data Consistency
................

Input data needs to have consistent types for ``idx``. Both string and numerical
types are allowed for ``idx``, but the original type and the referencing type
must be the same. Suppose we have a bus and a connected PQ. The Bus device may
use ``1`` or ``'1'`` as its ``idx``, as long as the PQ device uses the same
value for its ``bus`` parameter.

The ANDES xlsx reader will try to convert data into numerical types when
possible. This is especially relevant when the input ``idx`` is string literal
of numbers, the exported file will have them converted to numbers. The
conversion does not affect the consistency of data.

Parameter Check
...............
The following parameter checks are applied after converting input values to
array:

- Any ``NaN`` values will raise a ``ValueError``
- Any ``inf`` will be replaced with :math:`10^{8}`, and ``-inf`` will be
  replaced with :math:`-10^{8}`.


ANDES JSON
----------

Overview
........

JSON is a portable format for storing data. It has been used in several other
power system tools, including `PowerModels
<https://lanl-ansi.github.io/PowerModels.jl/stable/>`_, `Pandapower
<https://www.pandapower.org/>`_, and
`NREL-SIIP <https://github.com/nrel-siip>`_.
It must be noted that JSON files from these tools are not interoperable because
JSON only defines the data structure, not the data itself.

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
        }  //      <- Toggler ends
    ],
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

    ],   //        <-Bus ends
    ...  //        <- other models
    }    //        <- whole system ends

There are thirdparty tools for editing JSON files, but we still recommend to
convert files to ``xlsx`` for editing. The conversion can be readily done with

.. code:: bash

    andes run kundur_full.json -c xlsx


