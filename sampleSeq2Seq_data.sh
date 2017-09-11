#!/bin/bash

#PBS -q SINGLE
#PBS -N seq2seq_data
#PBS -l ncpus=1
#PBS -j oe

if [ "${PBS_O_WORKDIR}" != "" ];then
  cd ${PBS_O_WORKDIR}
  export PYTHON_EGG_CACHE=/work/$USER/tmp/.python-eggs
  export OMP_NUM_THREADS=1
fi
source ~/.bashrc


#rm -rf ../../data/chat/model/chat.dat
#rm -rf ../../data/chat/model/dict.dat

aprun python ./sampleSeq2Seq_data.py --mode chat --indir ../../data/chat/json/init100 --indict ../../data/chat/model/dict_init.dat --outfile ../../data/chat/model/chat_init.dat --outdict ../../data/chat/model/dict_init.dat
#aprun python ./sampleSeq2Seq_data.py --mode chat --indir ../../data/chat/json/rest1046 --indict ../../data/chat/model/dict.dat --outfile ../../data/chat/model/chat.dat --outdict ../../data/chat/model/dict.dat
#aprun python ./sampleSeq2Seq_data.py --indir ../../data/chat/nuc/ --indict ../../data/chat/model/dict.dat --outfile ../../data/chat/model/chat.dat --outdict ../../data/chat/model/dict.dat
