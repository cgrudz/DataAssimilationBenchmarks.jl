# GenerateTimeSeries 

GenerateTimeSeries is a sub-module used to generate a time series for a twin experiment based
on tuneable model configuration parameters. Example syntax for the configuration of a time
series is as follows, where arguments are defined in a 
[NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple)
to be passed to the
specific experiment function:
```{julia}
    (seed::Int64, h::Float64, state_dim::Int64, tanl::Float64, nanl::Int64, spin::Int64,
     diffusion::Float64)::NamedTuple)
```
Conventions for these arguments are as follows:
  * `seed` - specifies initial condition for the pseudo-random number generator on which various simulation settings will depend, and will be reproduceable with the same `seed` value;
  * `h` - is the numerical integration step size, controling the discretization error of the model evolution;
  * `state_dim` - controls the size of the [Lorenz-96 model](@ref) model though other models such as the [IEEE39bus](@ref) model are of pre-defined size;
  * `tanl` - (__time-between-analysis__)defines the length of continuous time units between sequential observations;
  * `nanl` - (__number-of-analyses__) defines the number of observations / analyses to be saved;
  * `spin` - discrete number of `tanl` intervals to spin-up for the integration of the dynamical system solution to guarantee a stationary observation generating process;
  * `diffusion` - determines intensity of the random perturbations in the integration scheme;

Results are saved in [.jld2 format](https://juliaio.github.io/JLD2.jl/dev/) in the data directory to be called by filter / smoother
experiments cycling over the pseudo-observations.

## Time series experiments

```@autodocs
Modules = [DataAssimilationBenchmarks.GenerateTimeSeries]
```
