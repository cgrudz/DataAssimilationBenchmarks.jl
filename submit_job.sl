#!/bin/bash
#SBATCH -n 1
#SBATCH -o experiment.out
#SBATCH -e experiment.err
julia SlurmExperimentDriver.jl "2"