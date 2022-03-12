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

.. image:: xlsx-bus.png
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

.. image:: xlsx-pq.png
   :width: 600
   :alt: Example workbook for PQ

In the above example PQ workbook, there are two PQ instances called ``PQ_0`` and
``PQ_1`` (referred to by ``idx``). They are connected to buses ``7`` and ``8``.
Therefore, on the ``Bus`` sheet, two rows need to exist with ``idx`` being ``7``
and ``8``.

Creating cases
..............

TODO

Adding workbooks
................

If one wants to add workbooks to an existing xlsx file, one can combine option
``--add-book ADD_BOOK`` (or ``-b ADD_BOOK``), where ``ADD_BOOK`` can be a single
model name or comma-separated model names (without any space). For example,

.. code:: bash

    andes run kundur.raw -c -b Toggler

will convert file ``kundur.raw`` into an ANDES xlsx file (kundur.xlsx) and add
a template workbook for `Toggler`.

.. Warning::
    With ``--add-book``, the xlsx file will be overwritten.
    Any **empty or non-existent models** will be REMOVED.

Autofill data
.............

TODO

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
