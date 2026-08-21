#!/bin/bash -l

#PBS -N CESM_metrics
#PBS -A P48500028
#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=16:mem=256GB
#PBS -q casper
#PBS -o CESM_metrics.log
#PBS -e CESM_metrics.err

conda activate credit
cd /glade/u/home/ksha/W2CC-base/verification/scripts/
python GLOBE_01_CESM_metrics.py

