.. _install:

*************************
Installation
*************************

ANDES can be installed in Python 3.6+.

Environment
===========

Setting Up Miniconda
--------------------
We recommend the Miniconda distribution that includes the conda package manager and Python.
Downloaded and install the latest Miniconda (x64, with Python 3)
from https://conda.io/miniconda.html.

Step 1: Open the Anaconda Prompt and create an environment for ANDES (optional)

.. code:: bash

     conda create --name andes python=3.7

Activate the new environment. On Microsoft Windows, do

.. code:: bash

     activate andes

On Linux or macOS, do

.. code:: bash

     conda activate andes


You can skip this step to install ANDES to the base environment, though it is not recommended.

Step 2: Add the ``conda-forge`` channel and set it as default

.. code:: bash

     conda config --add channels conda-forge
     conda config --set channel_priority flexible

Existing Python Environment (Advanced)
--------------------------------------
This is for advanced user only. Please skip it if you have set up a Conda envirnonment.
Instead of using Conda, if you prefer an existing Python environment,
you can install ANDES with `pip`:

.. code:: bash

      python3 -m pip install andes

If you see a `Permission denied` error, you will need to
install the packages locally with `--user`

Install ANDES
=============

ANDES can be installed in the user mode and the development mode.

User Mode
---------
If you want to use ANDES without modifying the source code, you can install it in the user mode.

In the Anaconda environment, run

.. code:: bash

    conda install andes


Developer Mode (Recommended)
----------------------------
If you want to hack into the code and, for example, develop new models or routines, please install it in the
development mode (recommended). The development mode has the same usage as the user mode.
In addition, changes to the source code will be reflected immediately without having to re-install the package.

Step 1: Get ANDES source code

Download the ANDES source code from
https://github.com/cuihantao/andes and extract all files to the path of your choice.
You can also ``git clone`` the source code (recommended).

.. code:: bash

    git clone https://github.com/cuihantao/andes

Step 2: Install dependencies

In the Anaconda environment, use ``cd`` to change directory to the ANDES root folder.

Install dependencies with

.. code:: bash

    conda install --file requirements.txt
    conda install --file requirements-dev.txt

Step 3: Install ANDES in the development mode using

.. code:: bash

      python3 -m pip install -e .

Pip will take care of the rest.

Optional Packages
-----------------

Install `cvxoptklu` to use KLU for speed up.
`cvxoptklu` is a standalone KLU direct solver for linear equations.
KLU is generally ~20% faster than UMFPACK.
cvxoptklu requires a C compiler, and the `openblas` and
`SuiteSparse` libraries.

.. code:: bash

      python3 -m install cvxoptklu

The installation of optional packages can be safely ignored and will not affect the functionality of ANDES.
