# ParallelExperimentDriver 

## ParallelExperimentDriver API

The `ParallelExperimentDriver.jl` is a simple parallel implementation of calling experiment parameter arrays
with `pmap` and Julia's native distributed computing.  This defines argumentless functions to construct
the parameter array and input data necessary to generate a sensitivity test, and implements a soft-fail for
experiments instability is encountered causing a crash of an experiment. This means that if a single experiment
configuration in the parameter array fails due to overflow, the remaining configurations will continue their
own course unaffected.

## Docstrings
#```@autodocs
#Modules = [DataAssimilationBenchmarks.ParallelExperimentDriver]
#```
