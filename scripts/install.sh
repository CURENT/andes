sudo apt update
sudo apt install build-essential git python3-dev libopenblas-dev suitesparse-dev

git submodule init && git submodule update

cd ../pkg/cvxopt
set CVXOPT_BLAS_LIB=openblas
set CVXOPT_LAPACK_LIB=openblas
python3 setup.py build
python3 setup.py install --user

cd ../..
python3 setup.py develop --user

andes andes/tests/ieee14_syn.dm -r t
andesplot ieee14_syn_out.dat 0 2 4 6
andes -c

