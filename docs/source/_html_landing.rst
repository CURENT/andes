**Python Software for Symbolic Power System Modeling and Numerical Analysis**

ANDES is an open-source Python library for power system modeling, computation, analysis, and control. It supports:

- **Power flow** calculation
- **Time-domain simulation** (transient stability)
- **Eigenvalue analysis** (small-signal stability)
- Symbolic-numeric framework for rapid model prototyping
- Full second-generation renewable energy models

----

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Complete learning path from installation through advanced analysis: power flow, time-domain simulation, eigenvalue analysis, parameter sweeps, and contingency studies.

   .. grid-item-card:: Modeling Guide
      :link: modeling/index
      :link-type: doc

      Framework internals: inspect model equations, understand the symbolic-numeric framework, and create new device models.

   .. grid-item-card:: Verification
      :link: verification/index
      :link-type: doc

      Validated against commercial tools including PSS/E and TSAT. Side-by-side comparison results demonstrate numerical accuracy for algorithms and models.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      CLI commands, configuration options, model reference (auto-generated), and API documentation.

----

Sponsors
--------

ANDES was initially developed at the `CURENT Engineering Research Center
<https://curent.utk.edu>`_ at the University of Tennessee, Knoxville, and has
been continually supported by startup funds and grants.

.. grid:: 1
   :gutter: 2
   :class-container: sponsor-logos

   .. grid-item::
      :class: sd-text-center

      .. image:: images/sponsors/CURENT_Logo_NameOnTrans.png
         :height: 80px
         :alt: CURENT

.. grid:: 2 3 5 5
   :gutter: 4

   .. grid-item::
      :class: sd-text-center

      .. image:: images/sponsors/nsf.jpg
         :height: 80px
         :alt: NSF

   .. grid-item::
      :class: sd-text-center

      .. image:: images/sponsors/doe.png
         :height: 80px
         :alt: DOE

   .. grid-item::
      :class: sd-text-center

      .. image:: images/sponsors/national_academies.png
         :height: 80px
         :alt: National Academies

   .. grid-item::
      :class: sd-text-center

      .. image:: images/sponsors/nlr.webp
         :height: 80px
         :alt: NLR

   .. grid-item::
      :class: sd-text-center

      .. image:: images/sponsors/nc_state.png
         :height: 80px
         :alt: NC State University

----

Useful Links
------------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item::

      .. button-link:: https://github.com/CURENT/andes/issues
         :color: primary
         :outline:
         :expand:

         :octicon:`issue-opened` Report Issues

   .. grid-item::

      .. button-link:: https://github.com/CURENT/andes/discussions
         :color: primary
         :outline:
         :expand:

         :octicon:`comment-discussion` Q&A

   .. grid-item::

      .. button-link:: https://mybinder.org/v2/gh/CURENT/andes/master
         :color: primary
         :outline:
         :expand:

         :octicon:`rocket` Try Online

   .. grid-item::

      .. button-link:: https://github.com/CURENT
         :color: primary
         :outline:
         :expand:

         :octicon:`repo` LTB Repository

----

Quick Install
-------------

.. tab-set::

   .. tab-item:: conda

      .. code-block:: bash

         conda install -c conda-forge andes

   .. tab-item:: pip

      .. code-block:: bash

         pip install andes

   .. tab-item:: uv

      .. code-block:: bash

         uv pip install andes

   .. tab-item:: development

      .. code-block:: bash

         git clone https://github.com/CURENT/andes
         cd andes
         pip install -e .[dev]

Quick Example
-------------

.. code-block:: python

   import andes

   # Load a test case and run power flow
   ss = andes.load(andes.get_case('ieee14/ieee14_fault.xlsx'))
   ss.PFlow.run()

   # Run time-domain simulation
   ss.TDS.run()

   # Plot generator speeds
   ss.TDS.plt.plot(ss.GENROU.omega)

----

Learning Paths
--------------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: New User

      1. :doc:`tutorials/01-installation`
      2. :doc:`tutorials/02-first-simulation`
      3. :doc:`tutorials/03-power-flow`

   .. grid-item-card:: Power System Analyst

      1. :doc:`tutorials/05-data-and-formats`
      2. :doc:`tutorials/07-eigenvalue-analysis`
      3. :doc:`tutorials/08-parameter-sweeps`

   .. grid-item-card:: Model Developer

      1. :doc:`tutorials/inspecting-models`
      2. :doc:`modeling/concepts/framework-overview`
      3. :doc:`modeling/creating-models/index`
