# SingleExperimentDriver 

## SingleExperimentDriver API

While the above filter experiments and smoother experiments configure twin experiments, run them and save the outputs,
the `SingleExperimentDriver.jl` and `ParallelExperimentDriver.jl` can be used as wrappers to run generic model settings for
debugging and validation, or to use built-in Julia parallelism to run a collection experiments over a parameter grid.
The `SingleExperimentDriver.jl` is primarily for debugging purposes with tools like `BenchmarkTools.jl` and `Debugger.jl`,
so that standard inputs can be run with the experiment called with macros.

## Docstrings
```@autodocs
Modules = [DataAssimilationBenchmarks.SingleExperimentDriver]
```
