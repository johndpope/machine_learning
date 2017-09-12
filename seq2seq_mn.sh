#!/bin/bash
#PBS -N seq2seq_mn
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



aprun -n 36 -N 36 python seq2seq_mn.py --dict ../../data/chat/model/dict_init.dat --data ../../data/chat/model/chat_init.dat --mode train --model ../../data/chat/model/seq_init/seq.model --out ../../data/chat/outdir  --communicator naive --stop 100e
