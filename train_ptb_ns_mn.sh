#!/bin/bash
#PBS -N ptb_mn_ns
#PBS -j oe 
#PBS -l select=1:ncpus=36:mpiprocs=36
#PBS -q SINGLE

# mpi

if [ "${PBS_O_WORKDIR}" != "" ];then
  cd ${PBS_O_WORKDIR}
  export PYTHON_EGG_CACHE=/work/$USER/tmp/.python-eggs
  export OMP_NUM_THREADS=1
fi
source ~/.bashrc


date
aprun -n 36 -N 36 python train_ptb_mn_ns.py --test True --out ../../data/machine_learning/train_ptb_mn_ns
date
