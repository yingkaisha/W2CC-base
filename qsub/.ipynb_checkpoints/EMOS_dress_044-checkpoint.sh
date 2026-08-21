#!/bin/bash -l

#PBS -N CSGD_dress
#PBS -A P48500028
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=32:mem=320GB
#PBS -q casper
#PBS -o EMOS_dress.log
#PBS -e EMOS_dress.err

conda activate credit
cd /glade/u/home/ksha/W2CC-base/EMOS/scripts/
python Dressing_QM.py

