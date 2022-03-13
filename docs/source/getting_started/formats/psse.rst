
.. _psse:

Siemens PSS/E
-------------

The Siemens PSS/E data format is a widely used for power system simulation.
PSS/E uses a variety of plain-text files to store data for different actions.
The RAW format (with file extension ``.raw``) is used to store the steady-state
data for power flow analysis, and the DYR format (with extension ``.dyr``) is
used to store the parameters of dynamic devices for transient simulation.

RAW Compatibility
.................
ANDES supports PSS/E RAW in versions 32 and 33. Newer versions of
``raw`` files can store PSS/E settings along with the system data, but such
feature is not yet supported in ANDES. Also, manually edited ``raw`` files can
confuse the parser in ANDES. Following manual edits, it is strongly recommended
to load the data into PSS/E and save the case as a v33 RAW file.

ANDES supports most power flow models in PSS/E. It needs to be recognized that
the power flow models in PSS/E is is a larger set compared with those in ANDES.
For example, switched shunts in PSS/E are converted to fixed ones, not all
three-winding transformer flags are supported, and HVDC devices are not yet
converted. This is not an exhaustive list, but all of them are advanced models.

We welcome contributions but please also reach out to us if you need
to arrange the development of such models.

DYR Compatibility
.................

Fortunately, the DYR format does not have different versions yet. ANDES support
reading parameters from DYR files for models that have been implemented in
ANDES. Owing to the descriptive modeling framework, we implement the identical
model so that parameters can be without conversion. If a dyr file contains
models that are not recognized by ANDES, an error will be thrown. One needs to
manually remove those unsupported models to load.

Like RAW files, manually edited DYR files can often be understood by PSS/E but
may confuse the ANDES parser. We also recommend to load and re-save the file
using PSS/E.

Loading files
.............

In the command line, PSS/E files can be loaded with

.. code-block:: bash

    andes run kundur.raw --addfile kundur.dyr

where ``--addfile`` or ``-a`` is used to specify the optional DYR file. For
now, DYR files can only be added to a RAW file. We will allow different formats
to be mixed in the future.

Likewise, one can convert PSS/E files to ANDES xlsx:

.. code-block:: bash

    andes run kundur.raw --addfile kundur.dyr -c

This will convert all models in the RAW and DYR files. If only the RAW file is
provided, only power flow models will be converted. One cannot easily append
those in a DYR file to an existing xlx file yet.

To load PSS/E files into a scripting environment, see Example - "Working with
Data".
