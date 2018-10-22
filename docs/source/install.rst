.. _install:

*************************
Installation instructions
*************************

Windows
=======
ANDES can be installed in Python 3.5+. We recommend the Miniconda distribution that includes
the conda package manager and Python. Downloaded and install the latest Miniconda (x64, with Python 3)
from https://conda.io/miniconda.html.

Open the Anaconda Prompt and create an environment for ANDES (optional) ::

     conda create --name andes python=3.6
     activate andes

Install ANDES using pip ::

     pip install andes

Alternative to installing as a distribution package, you can install ANDES in the development mode so that
your changes to the code will be reflected immesiately. Download the ANDES source code from 
https://github.com/cuihantao/Andes/releases, change the directory to the unzipped ANDES root folder and run ::

     pip install -e .

Pip should take care of the rest. 

macOS
=====
Install Python 3 using Miniconda (recommended) or Homebrew. Assume you have `python3` in your PATH, 
install ANDES with ::

     python3 -m pip install andes

Linux
=====

First make sure that you are using Python 3.5+. Miniconda is recommended but not required. 

Use your package manager to install ``blas``, ``lapack``, and ``SuiteSparse`` libraries.
For example, on Ubuntu, run::

     sudo apt-get install libblas-dev liblapack-dev libsuitesparse-dev

Install ANDES using ``pip``::

     python3 -m pip install andes

Install `cvxoptklu` to use KLU for speed up (optional, requires a C compiler) ::

     python3 -m pip install cvxoptklu

Installing from source
======================
**Meet the requirements**

*Required Python version*: 3.5/3.6

*Required Python packages*: CVXOPT, numpy, matplotlib and texttable.

*Optional Python packages*: sympy, cvxoptklu

Note: `pip` or `conda` package manager will take care of all the required packages.

**Download the ANDES source package**

The ANDES source code is available on GitHub at https://github.com/CURENT/Andes/releases.
You can either download the package and decompress it, or clone it with ``git``::

     git clone https://github.com/curent/andes.git

Open a command line terminal and navigate to the repository folder containing ``setup.py``,
install it as a development package using::

     python setup.py develop

**Install the optional packages via pip**

Some optional packages can be installed directly via ``pip``::

     pip install sympy

`cvxoptklu` is a standalone KLU direct solver for linear equations. KLU is generally
~20% faster than UMFPACK. cvxoptklu requires a C compiler, and the `openblas` and
`SuiteSparse` libraries.::

     pip install cvxoptklu

Troubleshooting
===============

There is a known issue of CVXOPT with versions earlier than 1.2.2 in Windows. If the time-domain
simulation crashes for the `cases/ieee14/ieee14_syn.dm`, please check and install the latest
CVXOPT.

