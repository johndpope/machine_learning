#!/bin/bash
#PBS -N copy_seq2seq
#PBS -j oe 
#PBS -l ncpus=1
#PBS -q SINGLE

if [ "${PBS_O_WORKDIR}" != "" ];then
  cd ${PBS_O_WORKDIR}
  export PYTHON_EGG_CACHE=/work/s1630401/tmp/.python-eggs
  export OMP_NUM_THREADS=1
fi
source ~/.bashrc

aprun python sampleCopySeq2Seq.py --dict ../../data/chat/txt/org/dict.dat --data ../../data/chat/txt/org/chat.dat --mode train --model ../../data/chat/txt/copy/copy.model  --epoch 100
#aws s3 cp --recursive ../../data/chat/txt  s3://kyodonews.advanced/utsubo/nlp/chat
#for f in *.txt;do
#  aws s3 cp $f  s3://kyodonews.advanced/utsubo/nlp/chat/
#done
#sudo shutdown -h now
