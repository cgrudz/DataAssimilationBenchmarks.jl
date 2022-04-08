# ParallelExperimentDriver 

## ParallelExperimentDriver API

The `ParallelExperimentDriver.jl` is a simple parallel implementation, though currently lacks a soft-fail when numerical
instability is encountered.  This means that if a single experiment configuration in the collection fails due to overflow, the entire
collection will cancel.  A fix for this is being explored, but the recommendation is to use the slurm submit
scripts below as templates for generating large parameter grid configurations and running them on servers.

## Docstrings
```@autodocs
Modules = [DataAssimilationBenchmarks.SingleExperimentDriver]
```
