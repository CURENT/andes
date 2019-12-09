.. _formats:

************************
I/O Formats
************************


==============================
Input Formats
==============================

ANDES currently supports the following input formats:

- ANDES Excel (.xlsx)
- MATPOWER (.m)
- Dome (.dm).


------------------------------
ANDES xlsx Format
------------------------------

The ANDES xlsx format is a newly introduced format since v0.7.0.
This format uses Microsoft Excel for conveniently viewing and editing model parameters.

Each worksheet in the xlsx file contains parameters for one model. The name of the worksheet
is the model name. The first rows in each woeksheet are the parameter field names.
Starting from the second row are the parameter for devices, each like representing one device.


Data Consistency
------------------------------

Input data needs to have consistent types for ``idx``. Both string and numerical types are allowed
for ``idx``, but the original type and the referencing type must be the same. For example,
suppose we have a bus and a connected PQ.
The Bus device may use ``1`` or ``'1'`` as its ``idx``, as long as the
PQ device uses the same value for its ``bus`` parameter.


The ANDES xlsx reader will try to convert data into numerical types when possible.
This means if the input ``idx`` is string literal of numbers, the exported file will have them
converted to numbers. The conversion does not affect the consistncy of data.

------------------------------
Parameter Check
------------------------------
The following parameter checks are applied after converting input values to array:

- Any ``NaN`` values will raise a ``ValueError``
- Any ``inf`` will be replaced with :math:`10^{10}`, and ``-inf`` will be replaced with :math:`-10^{10}`.


