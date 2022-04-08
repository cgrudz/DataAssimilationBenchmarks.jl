# Differential Equation Solvers


## API for differential equation solvers

Three general schemes are developed for ordinary and stochastic differential equations, the four-stage Runge-Kutta, second order Taylor,
and the Euler-(Maruyama) schemes.  Because the second order Taylor-Stratonovich scheme relies specifically on the structure of the
Lorenz-96 model with additive noise, this is included separately in the `models/L96.jl` sub-module.

General schemes such as the four-stage Runge-Kutta and the Euler-(Maruyama) schemes are built to take in arguments of the form

```{julia}
(x::VecA, t::Float64, kwargs::Dict{String,Any})
VecA = Union{Vector{Float64}, SubArray{Float64, 1}}

x            -- array or sub-array of a single state possibly including parameter values
t            -- time point
kwargs       -- this should include dx_dt, the paramters for the dx_dt and optional arguments
dx_dt        -- time derivative function with arguments x and dx_params
dx_params    -- ParamDict of parameters necessary to resolve dx_dt, not including those in the
                extended state vector
h            -- numerical discretization step size
diffusion    -- tunes the standard deviation of the Wiener process, equal to
                sqrt(h) * diffusion
diff_mat     -- structure matrix for the diffusion coefficients, replaces the default
                uniform scaling
state_dim    -- keyword for parameter estimation, dimension of the dynamic state < dimension
                of full extended state
param_sample -- ParamSample dictionary for merging extended state with dx_params
ξ            -- random array size state_dim, can be defined in kwargs to provide a
                particular realization of the Wiener process


# dictionary for model parameters
ParamDict = Union{Dict{String, Array{Float64}}, Dict{String, Vector{Float64}}}

# dictionary containing key and index pairs to subset the state vector
# and merge with dx_params
ParamSample = Dict{String, Vector{UnitRange{Int64}}}

```
with reduced options for the less-commonly used first order Euler(-Maruyama) scheme.

The time steppers over-write the value of `x` in place as a vector or a view of an array for efficient ensemble integration.
We follow the convention in data assimilation of the extended state formalism for parameter estimation where the parameter sample
should be included as trailing state variables in the columns of the ensemble array.  If

```{julia}
true == haskey(kwargs, "param_sample")
```
the `state_dim` parameter will specify the dimension of the dynamical states and create a view of the vector `x` including all entries
up to this index.  The remaining entries in the vector `x` will be passed to the `dx_dt` function in
a dictionary merged with the `dx_params` dictionary, according to the param_sample indices and parameter values specified in `param_sample`.
The parameter sample values will remain unchanged by the time stepper when the dynamical state entries in `x` are over-written in place.

Setting `diffusion > 0.0` above introduces additive noise to the dynamical system.  The main `rk4_step!` has convergence on order 4.0
when diffusion is equal to zero, and both strong and weak convergence on order 1.0 when stochasticity is introduced.  Nonetheless,
this is the recommended out-of-the-box solver for any generic DA simulation for the statistically robust performance, versus Euler-(Maruyama).
When specifically generating the truth-twin for the Lorenz-96 model with additive noise, this should be performed with the
`l96s_tay2_step!` in the `models/L96.jl` sub-module while the ensemble should be generated with the `rk4_step!`.
See the benchmarks on the [L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) for a full discussion of
statistically robust model configurations.

## Docstrings

```@autodocs
Modules = [DataAssimilationBenchmarks.DeSolvers]
```
