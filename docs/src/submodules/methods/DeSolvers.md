# Differential Equation Solvers

Three general schemes are developed for ordinary and stochastic differential equations,
  * the four-stage Runge-Kutta [`DataAssimilationBenchmarks.DeSolvers.rk4_step!`](@ref) scheme,
  * the second order autonomous Taylor [`DataAssimilationBenchmarks.DeSolvers.tay2_step!`](@ref) scheme, and
  * the Euler-(Maruyama) [`DataAssimilationBenchmarks.DeSolvers.em_step!`](@ref) scheme.

These schemes have arguments with the conventions:
  * `x` - model states of type [`VecA`](@ref) possibly including a statistical replicate of model parameter values;
  * `t` - time value of type [`Float64`](https://docs.julialang.org/en/v1/base/numbers/#Core.Float64) for present model state (a dummy argument is used for autonomous dynamics);
  * `kwargs` - a dictionary of type [`StepKwargs`](@ref).

Details of these schemes are available in the manuscript
[Grudzien et al. 2020](https://gmd.copernicus.org/articles/13/1903/2020/gmd-13-1903-2020.html)
Because the second order Taylor-Stratonovich scheme relies specifically on the structure
of the Lorenz-96 model with additive noise, this is included separately in the
[Lorenz-96 model](@ref) sub-module.  These time steppers over-write
the value of the model state `x` in-place for efficient ensemble integration.

The four-stage Runge-Kutta scheme follows the convention in data assimilation of the
extended state formalism for parameter estimation. In particular, the parameter sample
should be included as trailing state variables in the columns of the ensemble array.  If
the following conditional is true:
```{julia}
true == haskey(kwargs, "param_sample")
```
the `state_dim` parameter specifies the dimension of the dynamical states and creates a
view of the vector `x` including all entries up to this index.  The remaining entries in
the state vector `x` will be passed to the `dx_dt` function in
a dictionary merged with the `dx_params`  [`ParamDict`](@ref), according to the `param_sample`
indices and parameter values specified in `param_sample`. The parameter sample values
will remain unchanged by the time stepper when the dynamical state entries in `x` are
over-written in place.

Setting `diffusion > 0.0` introduces additive noise to the dynamical system.  The main
[`DataAssimilationBenchmarks.DeSolvers.rk4_step!`](@ref) has convergence on order 4.0
when diffusion is equal to zero, and both strong and weak convergence on order 1.0 when
stochasticity is introduced.  This is the recommended out-of-the-box solver for any
generic DA simulation for the statistically robust performance, versus Euler-(Maruyama).
When specifically generating the truth-twin for the Lorenz-96 model with additive noise,
this should be performed with the [`DataAssimilationBenchmarks.L96.l96s_tay2_step!`](@ref),
while the ensemble should be generated with the
[`DataAssimilationBenchmarks.DeSolvers.rk4_step!`](@ref).  See the benchmarks on the
[L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) for a full discussion of
statistically robust model configurations.

## Methods

```@autodocs
Modules = [DataAssimilationBenchmarks.DeSolvers]
```
