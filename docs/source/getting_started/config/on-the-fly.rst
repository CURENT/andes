Adjusting on the fly
--------------------

CLI
...
One can adjust config on the fly *in command line* without modifying or even
storing the config file. This is useful when the config change is one-time or
ANDES CLI is stored in a read-only container. The config update is done by
``andes run --config-option SECTION.OPTION=VALUE``, where ``SECTION`` is the
section name, ``OPTION`` is the option name, and ``VALUE`` is the new value.
*No space is allowed* around ``.`` and ``=``.

For example, to solve ``kundur_full.json`` with reactive power limit enforced,
one can do:

.. code:: bash

    andes run kundur_full.json -O PV.pv2pq=1

where ``-O`` is the shorhand command for ``--config-option``, and the enabled
``pv2pq`` will allow PV to be converted to PQ once reactive power limit is hit.

Multiple config updates can be passed simultaneously, separated by *space*. For
example, to enable reactive power limit and switch the power flow solver to
UMFPACK, do:

.. code:: bash

    andes run kundur_full.json -O PV.pv2pq=1 PFlow.sparselib=umfpack

Scripting
.........
To adjust config on the fly when scripting, there are two cases:

- Update the config when creating a new System object
- Update the config for an existing System object

To update the config when creating a new System object, one can pass a list of
strings to ``config_option`` for :py:mod:`andes.main.run`:

.. code:: python

    >>> ss = andes.run("kundur_full.json", config_option=["PV.pv2pq=1"])

which directly calls the backend API for the CLI. To update multiple configs,
one can do

.. code:: python

    >>> ss = andes.run("kundur_full.json",
                       config_option=["PV.pv2pq=1", "PFlow.sparselib=umfpack"])

When the System object gets created, the config values will be distributed to
member attributes of the System object. Therefore, the config for a System object
``ss`` is stored in ``ss.config``, and the config for the power flow routine is
stored in ``ss.PFlow.config``.

To update the config for an existing system, one can directly access the
``config`` attribute and set the new value. To set a new simulation end time,
one can overwrite the ``ss.TDS.config.tf`` field, such as:

.. code:: python

    # load system and run power flow
    >>> ss = andes.run("kundur_full.json")
    >>> ss.TDS.config.tf = 5.0
    >>> ss.TDS.run()

.. warning::

    Not all config options can be updated on the fly. Those config that are used
    for constructing the system object can only be updated when creating a new
    System object.
