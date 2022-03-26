.. ANDES documentation master file, created by
   sphinx-quickstart on Thu Jun 21 11:11:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================
ANDES documentation
===================


**Download documentation**: `PDF for stable version`_ | `PDF for development version`_

.. _`PDF for stable version`: https://docs.andes.app/_/downloads/en/stable/pdf/
.. _`PDF for development version`: https://docs.andes.app/_/downloads/en/latest/pdf/


**Useful Links**: `Binary Installer`_ | `Source Repository`_ | `Report Issues`_
| `Q&A`_ | `Try in Jupyter Notebooks`_

.. _`Source Repository`: https://github.com/cuihantao/andes
.. _`Report Issues`: https://github.com/cuihantao/andes/issues
.. _`Q&A`: https://github.com/cuihantao/andes/discussions
.. _`Binary Installer`: https://pypi.org/project/andes/
.. _`Try in Jupyter Notebooks`: https://mybinder.org/v2/gh/cuihantao/andes/master

ANDES is an open-source Python library for power system modeling, computation,
analysis, and control. It supports power flows calculation, transient stability
simulation, and small-signal stability analysis for transmission systems. ANDES
implements a symbolic-numeric framework for rapid prototyping of
differential-algebraic equation-based models. In this framework, a comprehensive
:ref:`library of models <modelref>` is developed, including the full
second-generation renewable models. Models in ANDES have been :ref:`verified
<verification>` with commercial software.

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---

    Getting started
    ^^^^^^^^^^^^^^^

    New to ANDES? Check out the getting started guides. They contain tutorials
    to the ANDES command-line interface, scripting usages, as well as guides to
    configure ANDES and work with case files.

    +++

    .. link-button:: getting-started
            :type: ref
            :text: To the getting started guides
            :classes: btn-block btn-secondary stretched-link

    ---

    Examples
    ^^^^^^^^

    The examples provide in-depth usage of ANDES in a Python scripting
    environment. Advanced usage and and power system studies are shown with
    explanation.

    +++

    .. link-button:: scripting_examples
            :type: ref
            :text: To the examples
            :classes: btn-block btn-secondary stretched-link

    ---

    Model development guide
    ^^^^^^^^^^^^^^^^^^^^^^^

    Looking to implement new models, algorithms and functionalities in ANDES?
    The development guide provides in-depth information on the design
    philosophy, data structure, and implementation of the hybrid
    symbolic-numeric framework.

    +++

    .. link-button:: development
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-secondary stretched-link
    ---

    API reference
    ^^^^^^^^^^^^^

    The API reference contains a detailed description of the ANDES package. The
    reference describes how the methods work and which parameters can be used.
    It assumes that you have an understanding of the key concepts.

    +++

    .. link-button:: api_reference
            :type: ref
            :text: To the API reference
            :classes: btn-block btn-secondary stretched-link

    ---
    :column: col-12 p-3

    Using ANDES for Research?
    ^^^^^^^^^^^^^^^^^^^^^^^^^
    Please cite our paper [Cui2021]_ if ANDES is used in your research for
    publication.


.. [Cui2021] H. Cui, F. Li and K. Tomsovic, "Hybrid Symbolic-Numeric Framework
       for Power System Modeling and Analysis," in IEEE Transactions on Power
       Systems, vol. 36, no. 2, pp. 1373-1384, March 2021, doi:
       10.1109/TPWRS.2020.3017019.


.. toctree::
   :caption: ANDES Manual
   :maxdepth: 3
   :hidden:

   getting_started/index
   examples/index
   modeling/index
   release-notes
   modelref
   api
