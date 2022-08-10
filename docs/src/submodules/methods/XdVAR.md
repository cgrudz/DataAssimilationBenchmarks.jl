# XdVAR

This module defines methods for classical variational data assimilation such as
3D- / 4D-VAR.  Primal cost functions are defined, with their implicit differentiation
performed with automatic differentiation with [JuliaDiff](https://github.com/JuliaDiff)
methods. Development of gradient-based optimization schemes using automatic
differentiation is ongoing, with future development planned to integrate variational
benchmark experiments.

The basic 3D-VAR cost function API is defined as follows
```{julia}
    D3_var_cost(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
```
where the control variable `x` is optimized, with fixed hyper-parameters defined in a
wrapping function passed to auto-differentiation.

## Methods

```@autodocs
Modules = [DataAssimilationBenchmarks.XdVAR]
```
