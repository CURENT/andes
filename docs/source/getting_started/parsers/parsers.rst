================
PSS/E Dyr Parser
================
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
