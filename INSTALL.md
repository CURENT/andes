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

Open ```powershell```, turn off ```Get-ExecutionPolicy``` and install [Chocolatey](https://chocolatey.org/install):
~~~~
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
iwr https://chocolatey.org/install.ps1 -UseBasicParsing | iex
choco install -y wget git 7zip.commandline
~~~~

Download and install Miniconda with Python 3.6 for your architecture from [Miniconda download page](https://conda.io/miniconda.html).
Switch to Python 3.4 and install required packages:
~~~~
conda install python=3.4
pip3 install numpy blist matplotlib texttable progressbar2 --user
~~~~

Run the following commands in the PowerShell to install the enhanced CVXOPT:
~~~~
wget http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-4.5.4.tar.gz
7z x SuiteSparse-4.5.4.tar.gz
7z x SuiteSparse-4.5.4.tar
set CVXOPT_SUITESPARSE_SRC_DIR=%CD%\SuiteSparse

wget https://bitbucket.org/carlkl/mingw-w64-for-python/downloads/OpenBLAS-0.2.17_amd64.7z
mkdir openblas
7z x OpenBLAS-0.2.17_amd64.7z -aoa -oopenblas

pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy

git clone https://github.com/sanurielf/cvxopt.git
cp openblas\amd64\lib\libopenblaspy.a cvxopt
cd cvxopt
set CVXOPT_BLAS_LIB=openblaspy
set CVXOPT_LAPACK_LIB=openblaspy
set CVXOPT_BLAS_LIB_DIR=%CD%
set CVXOPT_BLAS_EXTRA_LINK_ARGS=-lgfortran;-lquadmath
python setup.py build --compiler=mingw32
python setup.py install --user
python -m unittest discover -s tests
cd ..
~~~~

Clone the andes repository and install the development version:
~~~~
git clone https://github.com/cuihantao/andes.git
python3 setup.py develop --user
~~~~

FAQ:
1. Why do I have the error "DLL not loaded" while importing CVXOPT?

~~~
Try installing CVXOPT from a PowerShell with elevated privilege. 
Some links are not set correctly from Git Bash or cmd.
~~~