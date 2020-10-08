# ANDES

Python Software for Symbolic Power System Modeling and Numerical Analysis.

|               | Latest                                                                                                                                        | Stable                                                                                                                                        |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Documentation | [![Latest Documentation](https://readthedocs.org/projects/andes/badge/?version=latest)](https://andes.readthedocs.io/en/latest/?badge=latest) | [![Documentation Status](https://readthedocs.org/projects/andes/badge/?version=stable)](https://andes.readthedocs.io/en/stable/?badge=stable) |

| Badges        |                                                                                                                                                                                                                                                     |                                                                                                                                                                                                            |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Downloads     | [![PyPI Version](https://img.shields.io/pypi/v/andes.svg)](https://pypi.python.org/pypi/andes)                                                                                                                                                      | [![Conda Downloads](https://anaconda.org/conda-forge/andes/badges/downloads.svg)](https://anaconda.org/conda-forge/andes)                                                                                  |
| Try on Binder | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cuihantao/andes/master)                                                                                                                                                 |                                                                                                                                                                                                            |
| Code Quality  | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/17b8e8531af343a7a4351879c0e6b5da)](https://app.codacy.com/app/cuihantao/andes?utm_source=github.com&utm_medium=referral&utm_content=cuihantao/andes&utm_campaign=Badge_Grade_Dashboard) | [![Codecov Coverage](https://codecov.io/gh/cuihantao/andes/branch/master/graph/badge.svg)](https://codecov.io/gh/cuihantao/andes)                                                                          |
| Build Status  | [![GitHub Action Status](https://github.com/cuihantao/andes/workflows/Python%20application/badge.svg)](https://github.com/cuihantao/andes/actions)                                                                                                  | [![Azure Pipeline build status](https://dev.azure.com/hcui7/hcui7/_apis/build/status/cuihantao.andes?branchName=master)](https://dev.azure.com/hcui7/hcui7/_build/latest?definitionId=1&branchName=master) |

# Why ANDES
This software could be of interest to you if you are working on
DAE modeling, simulation, and control for power systems.
It has features that may be useful if you are applying
deep (reinforcement) learning to such systems.

ANDES is by far easier to use for developing differential-algebraic
equation (DAE) based models for power system dynamic simulation
than other tools such as
[PSAT](http://faraday1.ucd.ie/psat.html),
[Dome](http://faraday1.ucd.ie/dome.html) and
[PST](https://www.ecse.rpi.edu/~chowj/),
while maintaining high numerical efficiency.

ANDES comes with a rich set of commercial-grade dynamic models
with all details implemented, including limiters, saturation,
and zeroing out time constants.

ANDES produces credible simulation results. The following table
shows that

1. For the Northeast Power Coordinating Council (NPCC) 140-bus system
(with GENROU, GENCLS, TGOV1 and IEEEX1),
ANDES results match perfectly with that from TSAT.

2. For the Western Electricity Coordinating Council (WECC) 179-bus
system (with GENROU, IEEEG1, EXST1, ESST3A, ESDC2A, IEEEST and
ST2CUT), ANDES results match closely with those from TSAT and PSS/E.
Note that TSAT and PSS/E results are not identical, either.

|                                         NPCC Case Study                                                   |                                               WECC Case Study                                           |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-npcc/omega.png)   | ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-wecc/omega.png) |

ANDES provides a descriptive modeling framework in a scripting environment.
Modeling DAE-based devices is as simple as describing the mathematical equations.
Numerical code will be automatically generated for fast simulation.

| Controller Model and Equation | ANDES Code |
| ----------------------------- | ---------- |
| Diagram: <br> ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-tgov1/tgov1.png) <br><br> Write into DAEs: <br> ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-tgov1/tgov1_eqns.png)  | ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-tgov1/tgov1_class.png) |

In ANDES, what you simulate is what you document.
ANDES automatically generates model documentation, and the docs always stay up to date.
The screenshot below is the generated documentation for the implemented IEEEG1 model.

![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/misc/ieeeg1-screenshot.png)

In addition, ANDES features

+ a rich library of transfer functions and discontinuous components (including limiters, deadbands, and
  saturation functions) available for prototyping models, which can be effortlessly instantiated as multiple
  devices for system analysis
+ routines including Newton method for power flow calculation, implicit trapezoidal method for time-domain
  simulation, and full eigenvalue analysis
+ developed with performance in mind. While written in Python, ANDES comes with a performance package and can
  finish a 20-second transient simulation of a 2000-bus system in a few seconds on a typical desktop computer
+ out-of-the-box PSS/E raw and dyr data support for available models. Once a model is developed, inputs from a
  dyr file can be immediately supported

ANDES is currently under active development.
Use the following resources to get involved.

+ Start from the [documentation][readthedocs] for installation and tutorial.
+ Check out examples in the [examples folder][examples]
+ Read the model verification results in the [examples/verification folder][verification]
+ Try in Jupyter Notebook on [Binder][Binder]
+ Report bugs or issues by submitting a [GitHub issue][GitHub issues]
+ Submit contributions using [pull requests][GitHub pull requests]
+ Read release notes highlighted [here][release notes]
+ Check out and and cite our [paper][arxiv paper]

# Citing ANDES

If you use ANDES for research or consulting, please cite the following paper in your publication that uses
ANDES

```
H. Cui, F. Li and K. Tomsovic, "Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis,"
in IEEE Transactions on Power Systems, doi: 10.1109/TPWRS.2020.3017019.
```

# Who is Using ANDES?
Please let us know if you are using ANDES for research or projects.
We kindly request you to cite our [paper][arxiv paper] if you find ANDES useful.

![Natinoal Science Foundation](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/nsf.jpg)
![US Department of Energy](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/doe.png)
![CURENT ERC](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/curent.jpg)
![Lawrence Livermore National Laboratory](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/llnl.jpg)
![Idaho National Laboratory](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/inl.jpg)

# Sponsors and Contributors

This work was supported in part by the Engineering Research Center
Program of the National Science Foundation and the Department of Energy
under NSF Award Number EEC-1041877 and the CURENT Industry Partnership
Program.

See [GitHub contributors][GitHub contributors] for the contributor list.

# License

ANDES is licensed under the [GPL v3 License](./LICENSE).

* * *

[GitHub releases]:       https://github.com/cuihantao/andes/releases
[GitHub issues]:         https://github.com/cuihantao/andes/issues
[GitHub insights]:       https://github.com/cuihantao/andes/pulse
[GitHub pull requests]:  https://github.com/cuihantao/andes/pulls
[GitHub contributors]:   https://github.com/cuihantao/andes/graphs/contributors
[readthedocs]:           https://andes.readthedocs.io
[release notes]:         https://andes.readthedocs.io/en/latest/release-notes.html
[arxiv paper]:           https://arxiv.org/abs/2002.09455
[tutorial]:              https://andes.readthedocs.io/en/latest/tutorial.html#interactive-usage
[examples]:              https://github.com/cuihantao/andes/tree/master/examples
[verification]:          https://github.com/cuihantao/andes/tree/master/examples/verification
[Binder]:                https://mybinder.org/v2/gh/cuihantao/andes/master
