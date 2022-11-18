.. _numba-compilation:

Numba compilation
=================

.. _Numba: https://numba.pydata.org/

In nearly all numerical simulation software, time is mostly spent on
*constructing* the numerical system and *solving* it. The construction of the
DAE in ANDES involves the evaluation of functions from models that implement the
residuals and Jacobians.

Numba_ is a just-in-time compiler in Python that can turn numerical functions
into compiled machine code. In ANDES, it can speed up simulations by as much as
30%. The speedup is most effective in medium-sized systems with multiple
models. Such systems involve heavy function calls but rather moderate load for
linear equation solvers. It is is less significant in large-scale systems where
solving equations is the major time consumer.

.. note::

    Numba is supported since ANDES 1.5.0. One needs to manually install it with
    ``python -m pip install numba`` from the Anaconda Prompt.

Enabling Numba JIT
------------------

Numba needs to be enabled *manually*. In the ANDES config file: in section
``[System]``, set ``numba = 1``, so that it looks like ::

    [System]
       ...
       numba = 1
       ...

where the ``...`` are other options that are omitted here.

Just-in-time compilation will compile the code upon the *first execution* based
on the input types. This is the default mode of Numba. When compilation is
triggered, ANDES may appear frozen due to the compilation lag. To reuse the
compiled code and save compilation time for future runs, the compiled binary
code will be automatically cached. The default cache folder is in
``$HOME/.andes/pydata/__pycache__`` with file extensions ``nbc`` and ``nbi``

Numba compilation needs to be distinguished from the ANDES code generation by
:ref:`andes prepare`. The ANDES code generation is to generate Python code from
symbolically defined models and is relatively fast. The Numba compilation
further compiles the generated Python code to machine code. Whenever the ANDES
code generation produces new Python code, the cached Numba binary code will be
invalidated.

When not to compile
-------------------

when developing models, we recommend disabling numba to avoid spending time on
compilation.

Ahead-of-time compilation
-------------------------

Just-in-time compilation can feel laggy. When ANDES is not being developed, one
can compile the generated Python code ahead of time to avoid just-in-time
delays. We call it "precompilation".

Precompilation is invoked by

.. code:: bash

    andes prep -c

It may take a minute for the first time. Owing to caching, future compilations
will be incremental and much faster.