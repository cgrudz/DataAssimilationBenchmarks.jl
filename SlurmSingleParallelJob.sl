#!/bin/bash
#SBATCH --job-name=ParallelEnsembleRun
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=compute_bound
#SBATCH --mem-per-cpu=2500M
#SBATCH -t 12-00:00:00
#SBATCH -o ensemble.out
#SBATCH -e ensemble.err
#SBATCH --account=cpu-s1-ahn-0 
#SBATCH --partition=cpu-s1-ahn-0  
julia -p 31 ParallelExperimentDriver.jl

