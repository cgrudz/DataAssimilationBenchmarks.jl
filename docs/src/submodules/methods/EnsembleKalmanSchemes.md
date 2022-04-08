# Ensemble Kalman Schemes

## API for data assimilation solvers

Different filter and smoothing schemes are run through the routines including

```{julia}
ensemble_filter(analysis::String, ens::Array{Float64,2}, obs::Vector{Float64},
                         obs_cov::CovM, state_infl::Float64, kwargs::Dict{String,Any})

ls_smoother_classic(analysis::String, ens::Array{Float64,2}, obs::Array{Float64,2},
                             obs_cov::CovM, state_infl::Float64, kwargs::Dict{String,Any})

ls_smoother_single_iteration(analysis::String, ens::Array{Float64,2}, obs::Array{Float64,2},
                             obs_cov::CovM, state_infl::Float64, kwargs::Dict{String,Any})


ls_smoother_gauss_newton(analysis::String, ens::Array{Float64,2}, obs::Array{Float64,2},
                             obs_cov::CovM, state_infl::Float64, kwargs::Dict{String,Any};
                             ϵ::Float64=0.0001, tol::Float64=0.001, max_iter::Int64=10)

# type union for multiple dispatch over specific types of covariance matrices
CovM = Union{UniformScaling{Float64}, Diagonal{Float64}, Symmetric{Float64}}

analysis   -- string of the DA scheme string name, given to the transform sub-routine in
              methods
ens        -- ensemble matrix defined by the array with columns given by the replicates of
              the model state
obs        -- observation vector for the current analysis in ensemble_filter / array with
              columns given by the observation vectors for the ordered sequence of analysis
							times in the current smoothing window
obs_cov    -- observation error covariance matrix, must be positive definite of type CovM
state_infl -- multiplicative covariance inflation factor for the state variable covariance
              matrix, set to one this is the standard Kalman-like update
kwargs     -- keyword arguments for parameter estimation or other functionality, including
              integration parameters for the state model in smoothing schemes
```

The type of analysis to be passed to the transform step is specified with the `analysis` string.  Observations
for the filter schemes correspond to information available at a single analysis time while the ls (lag-shift)
smoothers require an array of observations corresponding to all analysis times within the DAW.  Observation
covariances should be typed according to the type union above for efficiency.  The state_infl is a required
tuneable parameter for multiplicative covariance inflation.   Extended parameter state covariance
inflation can be specified in `kwargs`.  These outer loops will pass the required values to the `transform` 
function that generates the ensemble transform for conditioning on observations.  Different outer-loop 
schemes can be built around the `transform` function alone in order to use validated ensemble transform 
schemes.  Utility scripts to generate observation operators, analyze ensemble statistics, etc, are included
in the EnsembleKalmanSchemes.jl sub-module.  See the experiments directory discussed below for example usage.


## Docstrings

```@autodocs
Modules = [DataAssimilationBenchmarks.EnsembleKalmanSchemes]
```
