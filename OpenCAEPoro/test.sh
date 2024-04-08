#!/bin/sh
#SBATCH -p G1Part_sce
#SBATCH -N 1
#SBATCH -n 56
#SBATCH --exclusive

# source /es01/paratera/parasoft/module.sh  
# source /es01/paratera/parasoft/oneAPI/2022.1/setvars.sh
# module load cmake/3.17.1 gcc/7.3.0-para
# unset I_MPI_PMI_LIBRARY
# export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

# Input np
read -p "Input the number of processes: " np
# Input verbose
read -p "Switch on verbose? (0/[1]): " verbose
# Choose test or full
read -p "Run full data? (yes/[no]): " full_data

if [ "$verbose" != "0" ]; then
	verbose=1
fi

if [ "$full_data" = "yes" ]; then
	path=./data/case1/case1.data
else
	path=./data/test/test.data
fi

echo "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "np: $np"
echo "verbose: $verbose"
echo "data path: $path"
echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"

# Run
mpirun -n $np ./testOpenCAEPoro $path verbose=$verbose

echo "Done!"
