#!/bin/bash

gpunum=$1
alpha1=$2
alpha2=$3

for i in 1 2 3
do
  echo $i
  /opt/anaconda3/bin/python -u decorrelation_Wjets_DNN_eval_v5.py --gpunum=$gpunum --decorr_mode='dist'  --alphamin=$alpha1 --alphamax=$alpha1 --nalpha=1 --logfile=log${alpha1}.csv --label=$i &> loglog_eval_${alpha1}_${i}
done

if [ "$alpha2" -ne "None" ]
then
for i in 1 2 3
do
  echo $i
  /opt/anaconda3/bin/python -u decorrelation_Wjets_DNN_eval_v5.py --gpunum=$gpunum --decorr_mode='dist'  --alphamin=$alpha2 --alphamax=$alpha2 --nalpha=1 --logfile=log${alpha2}.csv --label=$i &> loglog_eval_${alpha2}_${i}
done
fi
