
.. _psse:

PSS/E RAW and DYR
-----------------

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


.. _`creating disturbances`:

Creating disturbances
=====================
Instead of converting ``raw`` and ``dyr`` to ``xlsx`` before adding
disturbances, one can edit the ``.dyr`` file with a planin-text editor (such as
Notepad) and append lines customized for ANDES models. This is for advanced
users after referring to ``andes/io/psse-dyr.yaml``, at the end of which one can
find the format of ``Toggler``: ::

    # === Custom Models ===
    Toggler:
        inputs:
            - model
            - dev
            - t

To define two Togglers in the ``.dyr`` file, one can append lines to the end of
the file using, for example, ::

    Line   'Toggler'  Line_2  1 /
    Line   'Toggler'  Line_2  1.1 /

which is separated by spaces and ended with a slash. The second parameter is
fixed to the model name quoted by a pair of single quotation marks, and the
others correspond to the fields defined in the above ``inputs``. Each entry is
properly terminated with a forward slash.

Mapping dyr to ANDES models
===========================

ANDES supporting parsing PSS/E dynamic files in the format of ``.dyr``.
Support new dynamic models can be added by editing the input and output
conversion definition file in ``andes/io/psse-dyr.yaml``,
which is in the standard YAML format.
To add support for a new dynamic model, it is recommended to start with
an existing model of similar functionality.

Consider a ``GENCLS`` entry in a dyr file. The entry looks like ::

      1 'GENCLS' 1    13.0000  0.000000  /

where the fields are in the order of bus index, model name,
generator index on the bus, inertia (H) and damping coefficient (D).

The input-output conversion definition for GENCLS is as follows ::

    GENCLS:
        destination: GENCLS
        inputs:
            - BUS
            - ID
            - H
            - D
        find:
            gen:
                StaticGen:
                    bus: BUS
                    subidx: ID
        get:
            u:
                StaticGen:
                    src: u
                    idx: gen
            Sn:
                StaticGen:
                    src: Sn
                    idx: gen
            Vn:
                Bus:
                    src: Vn
                    idx: BUS
            ra:
                StaticGen:
                    src: ra
                    idx: gen
            xs:
                StaticGen:
                    src: xs
                    idx: gen
        outputs:
            u: u
            bus: BUS
            gen: gen
            Sn: Sn
            Vn: Vn
            D: D
            M: "GENCLS.H; lambda x: 2 * x"
            ra: ra
            xd1: xs

It begins with a base-level definition of the model name to be parsed from the
dyr file, namely, ``GENCLS``. Five directives can be defined for each model:
``destination``, ``inputs``, ``outputs``, ``find`` and ``get``.
Note that ``find`` and ``get`` are optional, but the other three are mandatory.

- ``destination`` is ANDES model to which the original PSS/E model will be
  converted. In this case, the ANDES model have the same name ``GENCLS``.
- ``inputs`` is a list of the parameter names for the PSS/E data.
  Arbitrary names can be used, but it is recommended to use the same notation
  following the PSS/E manual.
- ``outputs`` is a dictionary where the keys are the ANDES model parameter and
  the values are the input parameter or lambda functions that processes the inputs
  (see notes below).
- ``find`` is a dictionary with the keys being the temporary parameter name to store
  the ``idx`` of
  external devices and the values being the criteria to locate the devices.
  In the example above, ``GENCLS`` will try to find the ``idx`` of ``StaticGen``
  with ``bus == BUS`` and the ``subidx == ID``, where ``BUS`` and ``ID`` are from
  the dyr file.
- ``get`` is a dictionary with each key being a temporary parameter name for storing
  an external parameter and each value being the criteria to find the external parameter.
  In the example above, a temporary parameter ``u`` is the ``u`` parameter of ``StaticGen``
  whose ``idx == gen``. Note that ``gen`` is the ``idx`` of ``StaticGen`` retrieved
  in the above ``find`` section.

For the ``inputs`` section, one will need to skip the model name
because for any model, the second field is always the model name.
That is why for ``GENCLS`` below, we only list four input parameters. ::

    1 'GENCLS' 1    13.0000  0.000000  /

For the ``outputs`` section, the order can be arbitrary, but it is recommended
to follow the input order as much as possible for maintainability.
In particular, the right-hand-side of the outputs can be either an input parameter name
or an anonymous expression that processes the input parameters.
For the example of GENCLS, since ANDES internally uses the parameter of ``M = 2H``,
the input ``H`` needs to be multiplied by 2.
It is done by the following ::

    M: "GENCLS.H; lambda x: 2 * x"

where the left-hand-side is the output parameter name (destination ANDES model parameter name),
and the right-hand-side is arguments and the lambda function separated by semi-colon, all in a
pair of double quotation marks.
Multiple arguments are accepted and should be separated by comma.
Arguments can come from the same model or another model.
In the case of the same model, the model name can be neglected, namely, by writing
``M: "H; lambda x: 2 * x"``.
