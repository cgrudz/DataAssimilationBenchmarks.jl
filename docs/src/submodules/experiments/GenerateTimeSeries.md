# GenerateTimeSeries 

GenerateTimeSeries is a sub-module used to generate a time series for a twin experiment based
on tuneable model configuration parameters. Example syntax for the configuration of a time
series is as follows, where arguments are defined in a named tuple to be passed to the
specific experiment function:
```{julia}
    (seed::Int64, h::Float64, state_dim::Int64, tanl::Float64, nanl::Int64, spin::Int64,
     diffusion::Float64)::NamedTuple)
```
The `seed` argument specifies initial condition for the pseudo-random number generator
on which various simulation settings will depend -- simulations will be reproduceable with
the same `seed` value.  The numerical integration step size is given by the argument `h`
which controls the discretization error of the numerically simulated experiment.
The size of the Lorenz-96 system scales with the `state_dim` argument, though other
models such as the `IEEE39bus` model are of pre-defined size. The length of continuous
time units between sequential observations is controlled with the `tanl`
(time-between-analysis) argument which defines the frequency of pseudo data outputs.
The number of observations / analyses to be saved is controlled with the `nanl`
(number-of-analyses) argument.  The length of the spin-up for the integration of
the dynamical system solution to guarantee a stationary observation generating process
is controlled with the `spin` argument. The diffusion parameter determining the
intensity of the random perturbations is controlled with the `diffusion` argument.
Results are saved in .jld2 format in the data directory to be called by filter / smoother
experiments cycling over the pseudo-observations.

## Docstrings

```@autodocs
Modules = [DataAssimilationBenchmarks.GenerateTimeSeries]
```
