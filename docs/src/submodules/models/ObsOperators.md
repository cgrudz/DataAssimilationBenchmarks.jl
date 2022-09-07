# Observation Operators

The methods in this module define observation operators mapping the state model to
the observation space.  In current experiments, the observation operator is hard-coded
in the driver script with a statement
```
H_obs = alternating_obs_operator
```
defining the observation operator. The dimension of the observation and the nonlinear
transform applied can be controlled with the parameters of
[`DataAssimilationBenchmarks.ObsOperators.alternating_obs_operator`](@ref).

Additional observation models are pending,
following the convention where observation operators will be defined both for
vector arguments and multi-arrays using mutliple dispatch with the conventions:
```
function H_obs(x::VecA(T), obs_dim::Int64, kwargs::StepKwargs) where T <: Real
function H_obs(x::ArView(T), obs_dim::Int64, kwargs::StepKwargs) where T <: Real
```
allowing for the same naming to be used for single states, time series of states and
ensembles of states.


## Methods
```@autodocs
Modules = [DataAssimilationBenchmarks.ObsOperators]
```
