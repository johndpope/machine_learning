#!/bin/bash

#PBS -q G-SINGLE
#PBS -N seq2seq_data
#PBS -l select=1
#PBS -j oe

cd $PBS_O_WORKDIR

. ~/.bashrc
export PATH=${PBS_O_PATH}

export LDPATH=/work/opt/glibc-2.17/lib/ld-linux-x86-64.so.2:$LD_LIBRARY_PATH
export LDLIBS=/work/opt/glibc-2.17/lib:/work/opt/gcc485/lib64:/opt/cuda/8.0/lib64:/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
export PYTHON_PATH=/work/${USER}/python/py35

${PYTHON_PATH}/bin/python sampleSeq2Seq.py --dict ../../data/chat/txt/dict.dat --data ../../data/chat/txt/chat.dat --mode train -g 0 --epoch 1000 
