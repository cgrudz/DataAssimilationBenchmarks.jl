# GenerateTimeSeries

## GenerateTimeSeries

GenerateTimeSeries is a sub-module used to generate a time series for a twin experiment based on tuneable
configuration parameters.  Currently, this only includes the L96s model, with parameters defined as
```{julia}
l96_time_series(args::Tuple{Int64,Int64,Float64,Int64,Int64,Float64,Float64})
seed, state_dim, tanl, nanl, spin, diffusion, F = args
```
The `args` tuple includes the pseudo-random seed `seed`, the size of the Lorenz system `state_dim`, the length of continuous
time in between sequential observations / analyses `tanl`, the number of observations / analyses to be saved
`nanl`, the length of the warm-up integration of the dynamical system solution to guarantee an observation generating
process on the attractor `spin`, the diffusion parameter determining the intensity of the random perturbations `diffusion`
and the forcing parameter `F`.  This automates the selection of the correct time stepper for the truth twin when using
deterministic or stochastic integration.  Results are saved in .jld2 format in the data directory.

## Docstrings

```@autodocs
Modules = [DataAssimilationBenchmarks.GenerateTimeSeries]
```
