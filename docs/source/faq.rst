.. _faq:

**************************
Frequently Asked Questions
**************************

Modeling
========

Admittance matrix
-----------------

Q: Where to find the line admittance matrix?

A: ANDES does not build line admittance matrix for computing
line power injections. Instead, line power injections are
computed as vectors on the two line terminal. This approach
generalizes line as a power injection model.

Q: Without admittance matrix, how to switch out lines?

A: Lines can be switched out and in by using ``Toggler``.
See the example in ``cases/kundur/kundur_full.xlsx``.
One does not need to manually trigger a Jacobian matrix rebuild
because ``Toggler`` automatically triggers it using the new
connectivity status.
