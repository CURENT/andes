# ANDES

[![GitHub Action Status](https://github.com/cuihantao/andes/workflows/Python%20application/badge.svg)](https://github.com/cuihantao/andes/actions)
[![Azure Pipeline build status](https://dev.azure.com/hcui7/hcui7/_apis/build/status/cuihantao.andes?branchName=master)](https://dev.azure.com/hcui7/hcui7/_build/latest?definitionId=1&branchName=master)
[![Codecov Coverage](https://codecov.io/gh/cuihantao/andes/branch/master/graph/badge.svg)](https://codecov.io/gh/cuihantao/andes)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/17b8e8531af343a7a4351879c0e6b5da)](https://app.codacy.com/app/cuihantao/andes?utm_source=github.com&utm_medium=referral&utm_content=cuihantao/andes&utm_campaign=Badge_Grade_Dashboard)

[![PyPI Version](https://img.shields.io/pypi/v/andes.svg)](https://pypi.python.org/pypi/andes)
[![Conda Downloads](https://anaconda.org/conda-forge/andes/badges/downloads.svg)](https://anaconda.org/conda-forge/andes)
[![Documentation Status](https://readthedocs.org/projects/andes/badge/?version=latest)](https://andes.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cuihantao/andes/master)

A Python-based tool for symbolic power system modeling and numerical analysis.

# Why ANDES

ANDES is by far easier to use for modeling power system devices than other 
simulation tools such as 
[PSAT](http://faraday1.ucd.ie/psat.html),
[Dome](http://faraday1.ucd.ie/dome.html) and
[PST](https://www.ecse.rpi.edu/~chowj/),
while maintaining high numerical efficiency.

ANDES produces accurate simulation results.
For the Kundur's two-area system with GENROU, TGOV1 and EXDC2, ANDES produces almost identical 
(<1% discrepancy) results to that from DSATools TSAT™.   

| Generator Speed | Excitation Voltage |
| --------------- | ------------------ |
| ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-kundur/omega.png) | ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-kundur/efd.png) |

ANDES provides a descriptive modeling framework in a scripting environment.
Modeling DAE-based devices is as simple as describing the mathematical equations.

| Controller Model and Equation | ANDES Code |
| ----------------------------- | ---------- |
| Diagram: ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-tgov1/tgov1.png) <br><br> DAE: ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-tgov1/tgov1_eqns.png)  | ![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/example-tgov1/tgov1_class.png) |

In ANDES, what you simulate is what you document. 
ANDES automatically generates model documentation, and the docs always stay up to date.
The screenshot below is the generated documentation for the implemented TGOV1 model.

![](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/misc/doc-screenshot.png)

In addition, ANDES features

-   Power flow, trapezoidal method-based time domain simulation, and full eigenvalue analysis.
-   Support PSS/E raw and dyr inputs among other formats.
-   Symbolic DAE modeling and automated code generation for numerical simulation.
-   Numerical DAE modeling for cases when symbolic implementations are difficult.
-   Modeling library with common transfer functions and discontinuous blocks.
-   Automatic sequential and iterative initialization (experimental) for dynamic models.
-   Full equation documentation of supported DAE models.

ANDES is currently under active development.
Use the following resources to get involved.

+ Learn more about ANDES by reading the [documentation][readthedocs]
+ Report bugs or issues by submitting a [GitHub issue][GitHub issues]
+ Submit contributions using [pull requests][GitHub pull requests]
+ Read release notes highlighted [here][release notes]
+ Check out and and cite our [paper][arxiv paper]

# Table of Contents
* [Get Started with ANDES](#get-started-with-andes)
* [Run Simulations](#run-simulations)
   * [Step 1: Power Flow](#step-1-power-flow)
   * [Step 2: Dynamic Analysis](#step-2-dynamic-analyses)
   * [Step 3: Plot Results](#step-3-plot-results)
* [Configure ANDES](#configure-andes)
* [Format Converter](#format-converter)
   * [Input Converter](#input-converter)
   * [Output Converter](#output-converter)
* [Model Development](#model-development)
   * [Step 1: Define Parameters](#step-1-define-parameters)
   * [Step 2: Define Externals](#step-2-define-externals)
   * [Step 3: Define Variables](#step-3-define-variables)
   * [Step 4: Define Equations](#step-4-define-equations)
   * [Step 5: Define Initializers](#step-5-define-initializers)
   * [Step 6: Finalize](#step-6-finalize)
* [API Reference](#api-reference)
* [Who is Using ANDES?](#who-is-using-andes)

# Get Started with ANDES

ANDES is a Python package and needs to be installed. 
We recommend Miniconda if you don't insist on an existing Python environment.
Downloaded and install the latest **64-bit** Miniconda3 for your platform from <https://conda.io/miniconda.html>.

Step 1: (Optional) Open the *Anaconda Prompt* (shell on Linux and macOS) and create a new environment.

Use the following command in the Anaconda Prompt: 

``` bash
conda create --name andes python=3.7
```

Step 2: Add the `conda-forge` channel and set it to default. Do

``` bash
conda config --add channels conda-forge
conda config --set channel_priority flexible
```

Step 3: Activate the new environment

This step needs to be executed every time a new Anaconda Prompt or shell is open. 
At the prompt, do

``` bash
conda activate andes
```

Step 4: Download and Install ANDES

- Download the latest ANDES source code from <https://github.com/cuihantao/andes/releases>.

- Extract the package to a folder where source code resides. Try to avoid spaces in any folder name.

- Change directory to the ANDES root directory, which contains ``setup.py``. In the prompt, run the following
 commands in sequence.

```bash
conda install --file requirements.txt --yes
conda install --file requirements-dev.txt --yes
pip install -e .
```

Observe if any error is thrown. If not, ANDES is successfully installed in the development mode.

Step 5: Test ANDES

After the installation, run ``andes selftest`` and check if all tests pass.

# Run Simulations
ANDES can be used as a command-line tool or a library. 
The following explains the command-line usage, which comes handy to run studies. 

For a tutorial to use ANDES as a library, visit the [interactive tutorial][tutorial].

ANDES is invoked from the command line using the command `andes`.
Running `andes` without any input is equal to `andes -h` or `andes --help`, 
which prints out a preamble and help commands:

        _           _         | Version 0.8.3.post24+g8caf858a
       /_\  _ _  __| |___ ___ | Python 3.7.1 on Darwin, 04/06/2020 08:47:43 PM
      / _ \| ' \/ _` / -_|_-< |
     /_/ \_\_||_\__,_\___/__/ | This program comes with ABSOLUTELY NO WARRANTY.

    usage: andes [-h] [-v {10,20,30,40,50}]
                 {run,plot,misc,prepare,doc,selftest} ...

    positional arguments:
      {run,plot,misc,prepare,doc,selftest}
                            [run] run simulation routine; [plot] plot simulation
                            results; [doc] quick documentation; [prepare] run the
                            symbolic-to-numeric preparation; [misc] miscellaneous
                            functions.

    optional arguments:
      -h, --help            show this help message and exit
      -v {10,20,30,40,50}, --verbose {10,20,30,40,50}
                            Program logging level in 10-DEBUG, 20-INFO,
                            30-WARNING, 40-ERROR or 50-CRITICAL.

The first level of commands are chosen from `{run,plot,misc,prepare,selftest}`.
Each command contains a group of subcommands, which can be looked up by appending `-h` to the first-level command. 
For example, use `andes run -h` to look up the subcommands in `run`. 

`andes` has an option for the program verbosity level, controlled by `-v` or `--verbose`.
Accpeted levels are the same as in the `logging` module:
10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL.
To show debugging outputs, use `-v 10`.

## Step 1: Power Flow
Pass the path to the case file to `andes run` to perform power flow calculation. 
It is recommended to change directory to the folder containing the test case before running.

The Kundur's two-area system can be located under `andes/cases/kundur` with the name`kundur_full.xlsx`.
Locate the folder in your system and use `cd` to [change directory](https://en.wikipedia.org/wiki/Cd_(command)).
To run power flow calculation, do

```bash
andes run kundur_full.xlsx
```

Power flow reports will be saved to the directory where andes is called.
The power flow report, named `kundur_full_out.txt`, contains four sections:
- system statistics,
- ac bus and dc node data,
- ac line data,
- the initialized values of algebraic variables and state variables.

## Step 2: Dynamic Analyses
ANDES comes with two dynamic analysis routines: time-domain simulation and eigenvalue analysis.

Option `-r` or `-routine` is used to specify the routine,
followed by the routine name. 
Available routine names include `pflow,
tds, eig`.
 
- `pflow` is the default power flow calculation and can be omitted.
- `tds` is for time domain simulation.
- `eig` is for for eigenvalue analysis.

To run time-domain simulation for `kundur_full.xlsx` in the current directory, do

``` bash
andes run kundur_full.xlsx -r tds
```
Two output files, ``kundur_full_out.lst`` and ``kundur_full_out.npy`` will be created for variable names
and values, respectively.

Likewise, to run eigenvalue analysis for `kundur_full.xlsx`, use

``` bash
andes run kundur_full.xlsx -r eig
```

The eigenvalue report will be written in a text file named ``kundur_full_eig.txt``.

### PSS/E raw and dyr support
ANDES supports the PSS/E v32 raw and dyr files for power flow and dynamic studies.
Example raw and dyr files can be found in `andes/cases/kundur`.
To perform a time-domain simulation for `kundur_full.raw` and `kundur_full.dyr`, run

```bash
andes run kundur_full.raw --addfile kundur_full.dyr -r tds
```

where `--addfile` takes the dyr file. 
Please note that the support for dyr file is limited to the models available in ANDES.  

## Step 3: Plot Results
``andes plot`` is the command-line tool for plotting.
Currently, it only supports time-domain simulation data.
Three arguments are needed: file name, x-axis variable index, and y-axis variable index (or indices).

Variable indices can be looked up by opening the `kundur_full_out.lst` file as plain text.
Index 0 is always the simulation time. 

Multiple y-axis variable indices can be provided in eithers space-separated format or the Pythonic comma-separated
style.

To plot speed (omega) for all generators with indices 2, 8, 14, 20, either do

```bash
andes plot kundur_full_out.npy 0 2 8 14 20
```
or

```bash
andes plot kundur_full_out.npy 0 2:21:6
```

# Configure ANDES

ANDES uses a config file to set runtime configs for system, routines and models. 
The config file is loaded at the time when ANDES is invoked or imported.

At the command-line prompt, 

- ``andes misc --save`` saves all configs to a file. By default, it goes to ``~/.andes/andes.conf``.
- ``andes misc --edit`` is a shortcut for editing the config file. It takes an optional editor name. 

Without an editor name, the following default editor is used: 
- On Microsoft Windows, it will open up a notepad.
- On Linux, it will use the ``$EDITOR`` environment variable or use ``gedit`` by default.
- On macOS, the default is vim.

# Format Converter

## Input Converter
ANDES recognizes a few input formats (MATPOWER, PSS/E and ANDES xlsx) and can convert input to the `xlsx` format.
This function is useful when one wants to use models that are unique in ANDES.
 
- `andes run CASENAME.ext --convert` performs the conversion to `xlsx`, where `CASENAME.ext` is the full test
 case name.
- `andes run CASENAME.ext --convert-all` performs the conversion and create empty sheets for all supported models.
- `andes run CASENAME.xlsx --add-book ADD_BOOK`, where `ADD_BOOK` is the workbook name (the sane as the model
 name) to be added.
 
For example, to convert `wscc9.raw` in the current folder to the ANDES xlsx format, run 
 
```bash
andes run wscc9.raw --convert
```
The command will write the output to `wscc9.xlsx` in the current directory.
An additional `dyr` file can be included through `--addfile`, as shown in 
[Step 2: Dynamic Analysis](#step-2-dynamic-analyses).
Power flow models and dynamic models will be consolidated and written to a single xlsx file.

### Adding Model Template to an Existing xlsx File 
To add new models to an existing `xlsx` file, one needs to create new workbooks (shown tabs at the bottom),
`--add-book` can add model templates to an existing xlsx file.
To add models `GENROU` and `TGOV1` to the xlsx  file `wscc9.xlsx`, run

```bash
andes run wscc9.xlsx --add-book GENROU,TGOV1
```
Two workbooks named "GENROU" and "TGOV1" will appear in the new `wscc9.xlsx` file.

**Warning**: `--add-book` will *overwrite* the original file. 
All empty workbooks will be discarded.
It is recommended to make copies to backup your cases.

## Output Converter
The output converter is used to convert `.npy` output to a comma-separated (csv) file.
 
To convert, do `andes plot OUTPUTNAME.npy -c `, where `OUTPUTNAME.npy` is the file name of the simulation output.

For example, to convert `kundur_full_out.npy` (in the current directory) to a csv file, run

```bash
andes plot kundur_full_out.npy -c
```
The output will be written to `kundur_full_out.csv` in the current directory.

# Model Development
The steps to develop new models are outlined. 
New models will need to be written in Python and incorporated in the ANDES source code.
Models are placed under `andes/models` with a descriptive file name for the model type.

If a new file is created, import the building block classes at the top of the file 

```python
from andes.core.model import ModelData, Model
from andes.core.param import IdxParam, NumParam, ExtParam
from andes.core.var import Algeb, State, ExtAlgeb, ExtState
from andes.core.service import ConstService, ExtService
from andes.core.discrete import AntiWindupLimiter
```

The TGOV1 model will be used to illustrate the model development process.

## Step 1: Define Parameters
Create a class to hold parameters that will be loaded from the data file.
The class inherits from `ModelData`

```python

class TGOV1Data(ModelData):
    def __init__(self):
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            )
        self.R = NumParam(info='Speed regulation gain under machine base',
                          tex_name='R',
                          default=0.05,
                          unit='p.u.',
                          ipower=True,
                          )
        self.wref0 = NumParam(info='Base speed reference',
                              tex_name=r'\omega_{ref0}',
                              default=1.0,
                              unit='p.u.',
                              )

        self.VMAX = NumParam(info='Maximum valve position',
                             tex_name='V_{max}',
                             unit='p.u.',
                             default=1.2,
                             power=True,
                             )
        self.VMIN = NumParam(info='Minimum valve position',
                             tex_name='V_{min}',
                             unit='p.u.',
                             default=0.0,
                             power=True,
                             )

        self.T1 = NumParam(info='Valve time constant',
                           default=0.1,
                           tex_name='T_1')
        self.T2 = NumParam(info='Lead-lag lead time constant',
                           default=0.2,
                           tex_name='T_2')
        self.T3 = NumParam(info='Lead-lag lag time constant',
                           default=10.0,
                           tex_name='T_3')
        self.Dt = NumParam(info='Turbine damping coefficient',
                           default=0.0,
                           tex_name='D_t',
                           power=True,
                           )
```

Note that the example above has all the parameters loaded in one class. 
In practice, it is recommended to create a base class for common parameters and let `TGOV2Data` inherit from it.
See the code in `andes/models/governor.py` for the example. 

## Step 2: Define Externals
Next, another class to hold the non-parameter instances is created. 
The class inherits from `Model` and takes three positional arguments by the constructor.

The code below defines parameters, variables and services retrieved from external models (specifically
, generators).

```python
class TGOV1Model(Model):
    def __init__(self, system, config):
        self.Sn = ExtParam(src='Sn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='S_m',
                           info='Rated power from generator',
                           unit='MVA',
                           export=False,
                           )
        self.Vn = ExtParam(src='Vn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='V_m',
                           info='Rated voltage from generator',
                           unit='kV',
                           export=False,
                           )
        self.tm0 = ExtService(src='tm',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'\tau_{m0}',
                              info='Initial mechanical input')
        self.omega = ExtState(src='omega',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'\omega',
                              info='Generator speed',
                              unit='p.u.'
                              )
```
In addition, a service can be defined for the inverse of the gain

```python
        self.gain = ConstService(v_str='u / R',
                                 tex_name='G',
                                 )
```

## Step 3: Define Variables
First of all, the turbine governor output modifies the generator power input. Therefore, the generator input
variable should be retrieved by the governor. Next, internal variables can be defined.

```python
        # mechanical torque input of generators
        self.tm = ExtAlgeb(src='tm',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'\tau_m',
                           info='Mechanical power to generator',
                           )

        self.pout = Algeb(info='Turbine final output power',
                          tex_name='P_{out}',
                          )
        self.wref = Algeb(info='Speed reference variable',
                          tex_name=r'\omega_{ref}',
                          )
        
        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          )
        self.wd = Algeb(info='Generator under speed',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        )
        self.pd = Algeb(info='Pref plus under speed times gain',
                        unit='p.u.',
                        tex_name="P_d",
                        )

        self.LAG_x = State(info='State in lag transfer function',
                           tex_name=r"x'_{LAG}",
                           )
        self.LAG_lim = AntiWindupLimiter(u=self.LAG_x,
                                         lower=self.VMIN,
                                         upper=self.VMAX,
                                         tex_name='lim_{lag}',
                                         )
        self.LL_x = State(info='State in lead-lag transfer function',
                          tex_name="x'_{LL}",
                          )
        self.LL_y = Algeb(info='Lead-lag Output',
                          tex_name='y_{LL}',
                          )
```
 
## Step 4: Define Equations
Set up the equation associated with **each** variable.
Algebraic equations are in the form of `g(x, y) = 0`.
Differential equations are in the form of `f(x, y) = \dot{x}`. 

```python
        self.tm.e_str = 'u*(pout - tm0)'

        self.wref.e_str = 'wref0 - wref'    
        self.pref.e_str = 'tm0 * R - pref'
        self.wd.e_str = '(wref - omega) - wd'
        self.pd.e_str='(wd + pref) * gain - pd'

        self.LAG_x.e_str = 'LAG_lim_zi * (1 * pd - LAG_x) / T1'

        self.LL_x.e_str = '(LAG_x - LL_x) / T3'
        self.LL_y.e_str='T2 / T3 * (LAG_x - LL_x) + LL_x - LL_y'
        self.pout.e_str = '(LL_y + Dt * wd) - pout'
```

## Step 5: Define Initializers
Initializers are used to set up initial values for variables.
Initializers are evaluated in the same sequence as the declaration of variables.
Initializer evaluation results are set to the corresponding variable. 
Usually, only internal variables (`Algeb` and `State`) require initializers.

```python
        self.wref.v_str = 'wref0'
        self.pout.v_str = 'tm0'

        self.LL_y.v_str = 'LAG_x'
        self.LL_x.v_str = 'LAG_x'
        self.LAG_x.v_str = 'pd'

        self.pd.v_str = 'tm0'
        self.wd.v_str = '0'
        self.pref.v_str = 'tm0 * R'
```

Alternatively, equations and initializers can be passed to keyword arguments `e_str` and `v_str`, respectively
, of the corresponding instance.
 
## Step 6: Finalize
This step provides additional information on the model. 
The group to which the device belongs need to be specified, and the routine this model supports need to updated.

For example, TGOV1 belongs to the `TurbineGov` group, which is defined in `andes/models/group.py`.
TGOV1 participates in the time-domain simulation and is not involved in power flow. 
The snipet below is added to the constructor of `class TGOV1Model`.

```python
        self.group = 'TurbineGov'
        self.flags.update({'tds': True})
```

Next, a `TGOV1` class need to be created as the final class. It is a bit boilerplate as of the current
implementation.

```python
class TGOV1(TGOV1Data, TGOV1Model):
    def __init__(self, system, config):
        TGOV1Data.__init__(self)
        TGOV1Model.__init__(self, system, config)
``` 

One more step, the class needs to be added to the package `__init__.py` file to be loaded.
Edit `andes/models/__init__.py` and add to `non_jit` whose keys are the file names and values are the classes in
the file.
To add `TGOV1`, locate the line with key `governor` and add `TGOV1` to the value list so that it looks like

```python
non_jit = OrderedDict([
    # ...
    ('governor', ['TG2', 'TGOV1']),
    # ...
])
```

Finally, run `andes prepare` from the command-line to re-generate code for the new model. 

# API Reference
The official [documentation][readthedocs] explains the complete list of modeling components.
The most commonly used ones are highlighted in the following. 

# Who is Using ANDES?
Please let us know if you are using ANDES for research or projects. 
We kindly request you to cite our [paper][arxiv paper] if you find ANDES useful.

![Natinoal Science Foundation](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/nsf.jpg)
![US Department of Energy](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/doe.png)
![CURENT ERC](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/curent.jpg)
![Lawrence Livermore National Laboratory](https://raw.githubusercontent.com/cuihantao/andes/master/docs/source/images/sponsors/llnl.jpg)

# Contributors

This work was supported in part by the Engineering Research Center
Program of the National Science Foundation and the Department of Energy
under NSF Award Number EEC-1041877 and the CURENT Industry Partnership
Program.

This project was originally inspired by the book 
[Power System Modelling and Scripting](https://www.springer.com/gp/book/9783642136689)
by Prof. Federico Milano.

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
