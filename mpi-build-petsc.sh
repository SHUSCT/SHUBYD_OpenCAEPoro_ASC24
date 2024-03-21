#!/bin/bash


# source /es01/paratera/parasoft/module.sh  
# source /es01/paratera/parasoft/oneAPI/2022.1/setvars.sh
# module load cmake/3.17.1 gcc/7.3.0-para 

export CC=mpiicx
export CXX=mpiicpx

# users specific directory paths
export PARMETIS_DIR=/home/jamesnulliu/Projects/OpenCAEPoro/parmetis-4.0.3
export PARMETIS_BUILD_DIR=/home/jamesnulliu/Projects/OpenCAEPoro/parmetis-4.0.3/build/Linux-x86_64
export METIS_DIR=/home/jamesnulliu/Projects/OpenCAEPoro/parmetis-4.0.3/metis
export METIS_BUILD_DIR=/home/jamesnulliu/Projects/OpenCAEPoro/parmetis-4.0.3/build/Linux-x86_64
export PETSC_DIR=/home/jamesnulliu/Projects/OpenCAEPoro/petsc-3.19.3
export PETSC_ARCH=petsc_install
export PETSCSOLVER_DIR=/home/jamesnulliu/Projects/OpenCAEPoro/petsc_solver


export CPATH=/home/jamesnulliu/Projects/OpenCAEPoro/petsc-3.19.3/include/:$CPATH
export CPATH=/home/jamesnulliu/Projects/OpenCAEPoro/petsc-3.19.3/petsc_install/include/:/home/jamesnulliu/Projects/OpenCAEPoro/parmetis-4.0.3/metis/include:/home/jamesnulliu/Projects/OpenCAEPoro/parmetis-4.0.3/include:$CPATH
export CPATH=/home/jamesnulliu/Projects/OpenCAEPoro/lapack-3.11/CBLAS/include/:$CPATH



# install
rm -fr build; mkdir build; cd build;

echo "cmake -DUSE_PETSCSOLVER=ON -DUSE_PARMETIS=ON -DUSE_METIS=ON -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_BUILD_TYPE=Release .."
cmake -DUSE_PETSCSOLVER=ON -DUSE_PARMETIS=ON -DUSE_METIS=ON -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_BUILD_TYPE=Release ..

echo "make -j 32"
make -j 32

echo "make install"
make install
