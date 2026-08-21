#!/bin/bash -l

#PBS -N SMYLE
#PBS -A P48500028
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=12:mem=256GB
#PBS -q casper
#PBS -o SMYLE.log
#PBS -e SMYLE.err

conda activate credit
cd /glade/u/home/ksha/W2CC-base/verification/scripts/
python GLOBE_01_CESM_metrics.py

