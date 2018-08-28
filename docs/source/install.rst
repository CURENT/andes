.. _install:

*************************
Installation instructions
*************************

Windows
=======
Coming soon.

macOS
=====
Coming soon.

Linux
=====

Use your package manager to install ``blas``, ``lapack``, and ``SuiteSparse`` libraries.
For example, on Ubuntu, run::

     sudo apt-get install libblas-dev liblapack-dev libsuitesparse-dev

Install ANDES using ``pip``::

     pip install andes


Installing from source
======================
**Meet the requirements**

*Required Python version*: 3.4/3.5/3.6

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

