# Check if this script is run in super user mode
if [ "$EUID" -eq 0 ]; then 
  echo "Do not run this script as root."
  exit 1
fi

CURRENT_DIR=$(pwd)
mkdir -p $CURRENT_DIR/logs

add_line_to_file() {
  local line_to_add="$1"
  local file="$2"

  if grep -Fxq "$line_to_add" "$file"; then
    echo "\"$line_to_add\" is already in $file."
  else
    echo "$line_to_add" >> "$file"
    echo "Added \"$line_to_add\" to $file."
  fi
}


# # >>> Intel oneAPI >>>
# add_line_to_file "# >>> Intel oneAPI >>>" "$HOME/.bashrc"
# echo "Installing Intel OneAPI..."
# (
# sh ./l_BaseKit_p_2024.0.1.46_offline.sh -a -s --eula accept
# sh ./l_HPCKit_p_2024.0.1.38_offline.sh -a -s --eula accept
# ) >> logs/install_oneapi.log 2>&1

# add_line_to_file "source $HOME/intel/oneapi/setvars.sh --force" "$HOME/.bashrc"

# add_line_to_file "# <<< Intel oneAPI <<<" "$HOME/.bashrc"
# # <<< Intel oneAPI <<<

source $HOME/.bashrc

export CC=mpiicx CXX=mpiicpx FC=mpiifx F77=mpiifx F90=mpiifx

# >>> lapack >>>
echo "Installing lapack..."
(
cd lapack-3.11
make blaslib
make cblaslib
make lapacklib
make lapackelib
cd ..
) >> logs/install_lapack.log 2>&1
# <<< lapack <<<

# >>> parmetis >>>
echo "Installing parmetis..."
(
cd parmetis-4.0.3
bash -i build-parmetis.sh
cd ..
) >> logs/install_parmetis.log 2>&1
# <<< parmetis <<<

# >>> hypre >>>
echo "Installing hypre..."
(
cd hypre-2.28.0
bash -i build-hypre.sh
cd ..
) >> logs/install_hypre.log 2>&1
# <<< hypre <<<

# >>> petsc >>>
echo "Installing petsc..."
(
cd petsc-3.19.3
bash -i build-petsc.sh
cd ..
) >> logs/install_petsc.log 2>&1
# <<< petsc <<<

# >>> petsc_solver >>>
echo "Installing petsc_solver..."
(
cd petsc_solver
bash -i build-petscsolver.sh
cd ..
) >> logs/install_petsc_solver.log 2>&1
# <<< petsc_solver <<<

# >>> OpenCAEPoro >>>
echo "Installing OpenCAEPoro..."
(
cd OpenCAEPoro
bash -i mpi-build-petsc.sh
cd ..
) >> logs/install_OpenCAEPoro.log 2>&1
# <<< OpenCAEPoro <<<

echo "Installation done!"

