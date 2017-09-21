Installation
======

Supported environments:
--------
Andes runs on Python 3.4+ with CVXOPT, Numpy and Matplotlib. Windows, Linux and MacOS are all supported.


Ubuntu
--------
Install ```git```, ```python3-dev```, ```build-essential``` and necessary libraries:

~~~~
# apt update
# apt install build-essential git python3-dev
# apt install libopenblas-dev suitesparse-dev
~~~~

Clone the andes repository and an enhanced CVXOPT repository (by sanurielf):
~~~~
git clone https://github.com/cuihantao/andes.git
git clone https://github.com/sanurielf/cvxopt.git
~~~~

Install CVXOPT and Andes as follows:
~~~~
set CVXOPT_BLAS_LIB=openblas
set CVXOPT_LAPACK_LIB=openblas

cd cvxopt
python3 setup.py build
python3 setup.py install --user
cd ..

cd andes
python3 setup.py develop --user
~~~~

If you encounter any problem while installing CVXOPT, please read [CVXOPT installation instructions](http://cvxopt.org/install/index.html).
 
Windows
--------

Run ```powershell``` as Administrator, type in the following command:
~~~~
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
~~~~
Type in ```Y``` when asked.

Run the following commands in the same PowerShell window:
~~~~
iwr https://chocolatey.org/install.ps1 -UseBasicParsing | iex
choco install -y wget git 7zip.commandline
~~~~

Download and install Miniconda with Python 3.6 for your architecture from [Miniconda download page](https://conda.io/miniconda.html).
After installing, open the ```Anaconda Prompt```, clone the `Andes` repository with

~~~~
git clone https://github.com/cuihantao/andes
~~~~

Run the automatic installation script with the following and wait for the demo to show up.

~~~~
cd scripts
install.bat
~~~~

Congratulations!