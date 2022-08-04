.. _install:

************
Installation
************

New to Python
=============

.. _Setup Mambaforge:

Setting Up Mambaforge
---------------------
If you are new to Python and want to get started quickly, you can use
Mambaforge, which is a conda-like package manager configured with conda-forge.

Step 1:

Downloaded the latest Mambaforge for your platform from
https://github.com/conda-forge/miniforge#mambaforge.
Most users will use ``x86_64(amd64)`` for Intel and AMD processors.
Mac users with Apple Silicon should use ``arm64(Apple Silicon)``
for best performance.

Next, complete the Mambaforge installation on your system.

.. note::

    Mambaforge is a drop-in replacement for conda. If you have an existing
    conda installation, you can replace all following ``mamba`` commands
    with ``conda`` and achieve the same functionality.

    If you are using Anaconda or Miniconda on Windows, you should open
    ``Anaconda Prompt`` instead of ``Miniforge Prompt``.

Step 2:

Open Terminal (on Linux or maxOS) or `Miniforge Prompt` (on Windows, **not cmd!!**).
Make sure you are in a conda environment - you should see ``(base)`` prepended to the
command-line prompt, such as ``(base) C:\Users\username>``.

Create an environment for ANDES (recommended)

.. code:: bash

     mamba create --name andes python=3.8

Activate the new environment with

.. code:: bash

     mamba activate andes

.. note::

    You will need to activate the ``andes`` environment every time
    in a new Miniforge Prompt or shell.

If these steps complete without error, you now have a working Python environment.
See the commands at the top to :ref:`getting-started` ANDES.

.. _Install_extras:

Extra support package
=====================

Some ANDES features require extra support packages, which are not installed by
default. For example, to build the documentation, one will need to install
development packages. Other packages will be required for interoperability.

The extra support packages are specified in groups. The following group names
are supported, with descriptions given below:

- ``dev``: packages to support development such as testing and documentation
- ``interop``: packages to support interoperability of ANDES and other power
  systems tools.

.. note::

    Extra support packages are not supported by conda/mamba installation. One
    needs to install ANDES with ``pip``.

To install packages in the ``dev`` when installing ANDES, do:

.. code:: bash

    pip install andes[dev]

To install all extra packages, do:

.. code:: bash

    pip install andes[all]

One can also inspect the ``requirements-extra.txt`` to identify the packages
for manual installation.

.. _Develop Install:

Develop Install
===============

The development mode installation is for users who want to modify
the code and, for example, develop new models or routines.
The benefit of development mode installation is that
changes to source code will be reflected immediately without re-installation.

Step 1: Get ANDES source code

As a developer, you are strongly encouraged to clone the source code using ``git``
from either your fork or the original repository. Clone the repository with

.. code:: bash

    git clone https://github.com/cuihantao/andes

.. note::

    Replace the URL with yours to use your fork. With ``git``, you can later easily
    update the source code and perform version control.

Alternatively, you can download the ANDES source code from
https://github.com/cuihantao/andes and extract all files to the path of your
choice. Although works, this method is discouraged because tracking changes and
pushing back code edits will require significant manual efforts.

.. _`Step 2`:

Step 2: Install dependencies

In the Mambaforge environment, use ``cd`` to change directory to the ANDES root folder.
The folder should contain the ``setup.py`` file.

Install dependencies with

.. code:: bash

    mamba install --file requirements.txt
    mamba install --file requirements-extra.txt

Alternatively, you can install them with ``pip``:

.. code:: bash

    pip install -r requirements.txt
    pip install -r requirements-extra.txt

Step 3: Install ANDES in the development mode using

.. code:: bash

      python3 -m pip install -e .

Note the dot at the end. Pip will take care of the rest.

.. note::

    The ANDES version number shown in ``pip list``
    will stuck at the version that was intalled, unless
    ANDES is develop-installed again.
    It will not update automatically with ``git pull``.

    To check the latest version number, check the preamble
    by running the ``andes`` command or chek the output of
    ``python -c "import andes; print(andes.__version__)"``

.. note::

    ANDES updates may infrequently introduce new package
    requirements. If you see an ``ImportError`` after updating
    ANDES, you can manually install the missing dependencies
    or redo `Step 2`_.

.. note::

    To install extra support packages, one can append ``[NAME_OF_EXTRA]`` to
    ``pip install -e .``. For example, ``pip install -e .[interop]`` will
    install packages to support interoperability when installing ANDES in the
    development, editable mode.

Updating ANDES
==============

.. warning::

    If ANDES has been installed in the development mode using source code, you
    will need to use ``git`` or the manual approach to update the source code.
    In this case, Do not proceed with the following steps, as they will install
    a separate site-package installation on top of the development one.

Regular ANDES updates will be pushed to both ``conda-forge`` and Python package index.
It is recommended to use the latest version for bug fixes and new features.
We also recommended you to check the :ref:`ReleaseNotes` before updating to stay informed
of changes that might break your downstream code.

Depending you how you installed ANDES, you will use one of the following ways to upgrade.

If you installed it from mamba or conda, run

.. code:: bash

    conda install -c conda-forge --yes andes

If you install it from PyPI (namely, through ``pip``), run

.. code:: bash

    python3 -m pip install --yes andes


Uninstall Multiple Copies
=========================

A common mistake new users make is to have multiple copies of ANDES installed in
the same environment. This can happen when one previously installed ANDES in the
development mode but later ran ``conda install`` or ``python3 -m pip install``
to install the latest version. As a result, only the most recently installed
version will be accessible.

In this case, we recommend that you uninstall all version and reinstall only one
copy using your preferred mode. Uninstalling all copies can be done by calling
``conda remove andes`` and ``python3 -m pip uninstall andes``. The prompted path
will indicate the copy to be removed. One may need to run the two commands for a
couple of time until the package managers indicate that the ``andes`` package
can no longer be found.

Troubleshooting
===============

If you get an error message on Windows, reading ::

    ImportError: DLL load failed: The specified module could not be found.

It is a path issue of your Python. In fact, Python on Windows is so broken that
many people are resorting to WSL2 just for Python. Fixes can be convoluted, but
the easiest one is to install ANDES in a Conda/Mambaforge environment.
