# Ensemble Kalman Schemes

## API for ensemble Kalman schemes 
There are currently four families of ensemble Kalman estimators available in this package,
which define the outer-loop of the data assimilation cycle.  Particularly, these define
how the sequential data assimilation cycle will pass over a time series of observations,
with more details in the [SmootherExps](@ref) documents.

Ensemble filters only produce analyses forward-in-time.  The classic lag-shift smoother runsi
identically to the filter in its forecast and filter steps, but includes an additional retrospective
analysis to past ensemble states stored in memory.  The single iteration smoother follows
the same convention as the classic smoother, except in that new cycles are initiated from
a past, reanlyzed ensemble state.  The Gauss-Newton iterative smoothers are 4D smoothers,
which iteratively optimize the initial condition at the beginning of a data assimilation
cycle, and propagate this initial condition to initialize the subsequent cycle. A full
discussion of these methods can be found in
[Grudzien, et al. 2021](https://gmd.copernicus.org/preprints/gmd-2021-306/).

For each outer-loop method defining the data assimilation cycle, different types of analyses
can be specified within their arguments.  Likewise, these outer-loop methods require
arguments such as the ensemble state or the range of ensemble states to analyze, an
observation to assimilate or a range of observations to assimilate, as the observation
operator and observation error covariance and key word arguments for running the
underlying dynamical state model. Examples of the syntax are below:

```{julia}
ensemble_filter(analysis::String, ens::ArView(T), obs::VecA(T), obs_cov::CovM(T),
    kwargs::StepKwargs) where T <: Float64

ls_smoother_classic(analysis::String, ens::ArView(T), obs::ArView(T), obs_cov::CovM(T),
    kwargs::StepKwargs) where T <: Float64

ls_smoother_single_iteration(analysis::String, ens::ArView(T), obs::ArView(T),
    kwargs::StepKwargs) where T <: Float64

ls_smoother_gauss_newton(analysis::String, ens::ArView(T), obs::ArView(T), obs_cov::CovM(T),
    kwargs::StepKwargs; ϵ::Float64=0.0001, tol::Float64=0.001,
    max_iter::Int64=10) where T <: Float64
```
with conventions defined as follows:
  * `analysis` - string name of the analysis scheme;
  * `ens` - ensemble matrix defined by the array with columns given by the replicates of the model state;
  * `obs` - observation vector for the current analysis in `ensemble_filter` / array with columns given by the observation vectors for the ordered sequence of analysis times in the current smoothing window;
  * `H_obs` - observation model mapping state vectors and ensembles into observed variables;
  * `obs_cov` - observation error covariance matrix;
  * `kwargs` - keyword arguments for inflation, parameter estimation or other functionality, including integration parameters for the state model in smoothing schemes.

The `analysis` string is passed to the
[`DataAssimilationBenchmarks.EnsembleKalmanSchemes.transform_R`](@ref) or the 
[`DataAssimilationBenchmarks.EnsembleKalmanSchemes.ens_gauss_newton`](@ref)
methods below to produce a specialized analysis within the outer-loop controlled by the above
filter and smoother methods. Observations for the filter
schemes correspond to information available at a single analysis time giving an observation
of the state vector of type [`VecA`](@ref). The `ls` (lag-shift) smoothers require an array of
observations of type [`ArView`](@ref) corresponding to all analysis times within the data
assimilation window (DAW). Observation covariances are typed as [`CovM`](@ref) for
efficiency.  State covariance multiplicative inflation and extended state parameter
covariance multiplicative inflation can be specified in `kwargs`.
Utility scripts to generate observation operators, analyze ensemble statistics, etc,
are included in the below. 

## Methods

```@autodocs
Modules = [DataAssimilationBenchmarks.EnsembleKalmanSchemes]
```
