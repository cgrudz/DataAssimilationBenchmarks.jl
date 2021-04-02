#!/bin/bash
#SBATCH --job-name=ParallelEnsembleRun
#SBATCH -n 32
#SBATCH -t 12-00:00:00
#SBATCH --mem-per-cpu=5000M
#SBATCH -t 12-00:00:00
#SBATCH -o ensemble.out
#SBATCH -e ensemble.err
#SBATCH --account=cpu-s1-ahn-0 
#SBATCH --partition=cpu-s1-ahn-0  
julia -p 31 ParallelExperimentDriver.jl

