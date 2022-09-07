# SingleExperimentDriver 

Following the convention of [GenerateTimeSeries](@ref), [FilterExps](@ref)
and [SmootherExps](@ref) using
[NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple)
arguments to  define hyper-parameter configurations,
the SingleExperimentDriver module defines dictionaries of the form
```
experiment_group["parameter_settings"]
```
where keyword arguments return standard parameter configurations for these experiments
with known results for reproducibility. These standard configurations are used in the package
for for debugging, testing, benchmarking and profiling code.  Package tests
use these standard configurations to verify a DA method's forecast and analysis RMSE.
User-defined, custom experiments can be modeled from the methods in the above modules with a
corresponding SingleExperimentDriver dictionary entry used to run and debug the experiment,
and to test and document the expected results. Parallel submission scripts are used
for production runs of sensitivity experiments, defined in [ParallelExperimentDriver](@ref).

## Experiment groups
```@autodocs
Modules = [DataAssimilationBenchmarks.SingleExperimentDriver]
```
