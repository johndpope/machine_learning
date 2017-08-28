#!/bin/bash
#PBS -N attseq2seq
#PBS -j oe 
#PBS -l ncpus=1
#PBS -q SINGLE

if [ "${PBS_O_WORKDIR}" != "" ];then
  cd ${PBS_O_WORKDIR}
  export PYTHON_EGG_CACHE=/work/s1630401/tmp/.python-eggs
  export OMP_NUM_THREADS=1
fi
source ~/.bashrc

aprun python sampleAttSeq2Seq.py --dict ../../data/chat/txt/dict.dat --data ../../data/chat/txt/chat.dat --model ../../data/chat/txt/att/att.model --mode train  --epoch 100 
#aws s3 cp --recursive ../../data/chat/txt  s3://kyodonews.advanced/utsubo/nlp/chat
#for f in *.txt;do
#  aws s3 cp $f  s3://kyodonews.advanced/utsubo/nlp/chat/
#done
#if [ -f att.model ];then
#  echo "sudo shutdown -h now"
##  sudo shutdown -h now
#fi
