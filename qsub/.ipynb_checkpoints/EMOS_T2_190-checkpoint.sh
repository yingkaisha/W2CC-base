#!/bin/bash -l

#PBS -N EMOS_T2
#PBS -A P48500028
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=12:mem=64GB
#PBS -q casper
#PBS -o EMOS_T2.log
#PBS -e EMOS_T2.err

conda activate credit
cd /glade/u/home/ksha/W2CC-base/EMOS/scripts/
python EMOS_coef.py 190

