#!/bin/bash
#SBATCH -n 1
#SBATCH -o hybrid_smoother.out
#SBATCH -e hybrid_smoother.err
julia SlurmExperimentDriver.jl "1815"