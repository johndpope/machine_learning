#!/bin/bash
#PBS -N seq2seq
#PBS -j oe 
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -q SINGLE

if [ "${PBS_O_WORKDIR}" != "" ];then
  cd ${PBS_O_WORKDIR}
  export PYTHON_EGG_CACHE=/work/$USER/tmp/.python-eggs
  export OMP_NUM_THREADS=36
fi
source ~/.bashrc

aprun -n 1 -d $OMP_NUM_THREADS python sampleSeq2Seq.py --dict ../../data/chat/model/dict.dat --data ../../data/chat/model/chat.dat --mode train --model ../../data/chat/model/seq/seq.model  --epoch 100
#aws s3 cp --recursive ../../data/chat/txt  s3://kyodonews.advanced/utsubo/nlp/chat
#for f in *.txt;do
#  aws s3 cp $f  s3://kyodonews.advanced/utsubo/nlp/chat/
#done
#sudo shutdown -h now
