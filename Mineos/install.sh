#!/bin/sh
current_dir=$(pwd)
gfortran -c minos_bran.f
./configure --prefix=$(pwd)/cig F77=gfortran
make
make install
