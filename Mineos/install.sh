#!/bin/sh
gfortran -c minos_bran.f
./configure --prefix=$HOME/cig F77=gfortran
make
make install
