#!/bin/bash
#PBS -N copy_seq2seq
#PBS -j oe 
#PBS -l select=1:ncpus=36:ompthreads=36
#PBS -q SINGLE

if [ "${PBS_O_WORKDIR}" != "" ];then
  cd ${PBS_O_WORKDIR}
  export PYTHON_EGG_CACHE=/work/$USER/tmp/.python-eggs
  export OMP_NUM_THREADS=36
fi
source ~/.bashrc

aprun -n 1 -d $OMP_NUM_THREADS python sampleCopySeq2Seq.py --dict ../../data/chat/model/dict.dat --data ../../data/chat/model/chat.dat --mode train --model ../../data/chat/model/copy/copy2.model  --epoch 300 --batch 50
#aws s3 cp --recursive ../../data/chat/txt  s3://kyodonews.advanced/utsubo/nlp/chat
#for f in *.txt;do
#  aws s3 cp $f  s3://kyodonews.advanced/utsubo/nlp/chat/
#done
#sudo shutdown -h now
