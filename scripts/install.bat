:: Conda and Python packages
conda install python=3.4 -y
pip install numpy texttable matplotlib progressbar2 sympy
pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy
pip install https://bitbucket.org/carlkl/mingw-w64-for-python/downloads/scipy-0.17.0-cp34-cp34m-win_amd64.whl

:: SuiteSparse
wget http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-4.5.4.tar.gz
7z x SuiteSparse-4.5.4.tar.gz
7z x SuiteSparse-4.5.4.tar
set CVXOPT_SUITESPARSE_SRC_DIR=%CD%\SuiteSparse
del SuiteSparse-4.5.4.tar.gz SuiteSparse-4.5.4.tar

:: OpenBLAS
wget https://bitbucket.org/carlkl/mingw-w64-for-python/downloads/OpenBLAS-0.2.17_amd64.7z
mkdir openblas
7z x OpenBLAS-0.2.17_amd64.7z -aoa -oopenblas
del OpenBLAS-0.2.17_amd64.7z

:: CVXOPT with KLU
git clone https://github.com/sanurielf/cvxopt.git
copy openblas\amd64\lib\libopenblaspy.a cvxopt
cd cvxopt
set CVXOPT_BLAS_LIB=openblaspy
set CVXOPT_LAPACK_LIB=openblaspy
set CVXOPT_BLAS_LIB_DIR=%CD%
set CVXOPT_BLAS_EXTRA_LINK_ARGS=-lgfortran;-lquadmath
python setup.py build --compiler=mingw32
python setup.py install
python -m unittest discover -s tests
cd ..

:: Andes develop 

cd ..
python setup.py develop

:: Andes test
andes andes/tests/ieee14_syn.dm -r t
andesplot ieee14_syn_out.dat 0 2 4 6
andes -c