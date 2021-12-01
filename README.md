# DataAssimilationBenchmarks.jl

![DataAssimilationBenchmarks.jl logo](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/blob/master/assets/dabenchmarks.png)

[![DOI](https://zenodo.org/badge/268903920.svg)](https://zenodo.org/badge/latestdoi/268903920)
[![Total lines of code without comments](https://tokei.rs/b1/github/cgrudz/DataAssimilationBenchmarks.jl?category=code)](https://github.com/cgrudz/DataAssimilationBenchmarks.jl)
[![codecov](https://codecov.io/gh/cgrudz/DataAssimilationBenchmarks.jl/branch/master/graph/badge.svg?token=3XLYTH8YSZ)](https://codecov.io/gh/cgrudz/DataAssimilationBenchmarks.jl)

## Welcome to DataAssimilationBenchmarks.jl!

### Description
This is my personal data assimilation benchmark research code with an emphasis on testing and validation
of ensemble-based filters and sequential smoothers in chaotic toy models.  The code is meant to be performant, 
in the sense that large hyper-parameter discretizations can be explored to determine structural sensitivity 
and reliability of results across different experimental regimes, with parallel implementations in Slurm.
This includes code for developing and testing data assimilation schemes in the 
[L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) currently, with further models in development.
This project supported the development of all numerical results and benchmark simulations considered in the pre-print
[A fast, single-iteration ensemble Kalman smoother for sequential data assimilation](https://gmd.copernicus.org/preprints/gmd-2021-306/)
available currently in open review in Geoscientific Model Development.

Lines of code counter (without comments or blank lines) courtesy of [Tokei](https://github.com/XAMPPRocky/tokei).

### Validated methods currently in use

<table>
<tr>
	<th>Estimator / implemented techniques</th>
	<th>Tuned multiplicative inflation</th>
	<th>Adaptive inflation, finite-size formalism (perfect model dual / primal)</th>
	<th>Adaptive inflation, finite-size formalism (imperfect model)</th>
	<th>Linesearch</th>
	<th>Localization / Hybridization</th>
	<th>Multiple data assimilation (general shift and lag)</th>
</tr>
<tr>
  <td>EnKF, perturbed obs.</td><td>X</td><td>X</td><td></td><td>NA</td><td></td><td>NA</td>
</tr>
<tr>
  <td>ETKF</td><td>X</td><td>X</td><td></td><td>NA</td><td></td><td>NA</td>
</tr>
<tr>
  <td>MLEF, transform / bundle variants</td><td>X</td><td>X</td><td></td><td>X</td><td></td><td>NA</td>
</tr>
<tr>
  <td>EnKS, perturbed obs.</td><td>X</td><td>X</td><td></td><td>NA</td><td></td><td>NA</td>
</tr>
<tr>
  <td>ETKS</td><td>X</td><td>X</td><td></td><td>NA</td><td></td><td>NA</td>
</tr>
<tr>
  <td>MLES, transform / bundle variants</td><td>X</td><td>X</td><td></td><td>X</td><td></td><td>NA</td>
</tr>
<tr>
  <td>SIEnKS, perturbed obs / ETKF / MLEF variants</td><td>X</td><td>X</td><td></td><td>X</td><td></td><td>X</td>
</tr>
<tr>
  <td>Gauss-Newton IEnKS, transform / bundle variants</td><td>X</td><td>X</td><td></td><td></td><td></td><td>X</td>
</tr>
</table>

### Structure
The directory is structured as follows:
<ul>
  <li> src - contains the main wrapper module</li>
  <ul>
		<li> models - contains code for defining the dynamic model equations in twin experiments.</li>
		<li> methods - contains DA solvers and general numerical routines for running twin experiments.</li>
		<li> experiments - contains the outer-loop scripts that set up twin experiments.</li>
		<li> data - this is an input / output directory for the inputs to and ouptuts from experiments.</li>
		<li> analysis - contains auxilliary scripts for batch processing experiment results and for plotting in Python.</li>
	</ul>
  <li> scratch - this is a storage directory for backups.</li>
  <li> test - contains test cases for the package, currently under development.</li>
</ul>

## Installation

The main module DataAssimilationBenchmarks.jl is a wrapper module including the core numerical solvers 
for ordinary and stochastic differential equations, solvers for data assimilation routines and the core 
process model code for running twin experiments with benchmark models. These methods can be run 
stand-alone in other programs by calling these functions from the DeSolvers, EnsembleKalmanSchemes and 
L96 sub-modules from this library. Future solvers and models will be added as sub-modules in the methods
and models directories respectively. 

In order to get the full functionality of this package you you will
need to install the dev version.  This provides the access to edit all of the outer-loop routines for 
setting up twin experiments. These routines are defined in the modules in the "experiments" directory.
The "slurm_submit_scripts" directory includes routines for parallel submission of experiments in Slurm.
Data processing scripts and visualization scripts (written in Python with Matplotlib and Seaborn) are 
included in the "analysis" directory.

### Installing a dev package from the Julia General registries 

In order to install the dev version to your Julia environment, you can use the following commands in the REPL

```{julia}
pkg> dev DataAssimilationBenchmarks
```

The installed version will be included in your

```
~/.julia/dev/
```
on Linux and the analogous directory with respect Windows and Mac systems.

Alternatively, you can install this from my Github directly as follows:
```{julia}
pkg> dev https://github.com/cgrudz/DataAssimilationBenchmarks.jl
```

## Using solvers in DataAssimilationBenchmarks

The methods directory currently contains two types of solvers, differential equation solvers in the DeSolvers.jl sub-module,
and ensemble-Kalman-filter-based data assimilation routines and utilities in the EnsembleKalmanSchemes.jl sub-module.

### API for differential equation solvers

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
dx_params    -- ParamDict of parameters necessary to resolve dx_dt, not including those in the extended state vector
h            -- numerical discretization step size
diffusion    -- tunes the standard deviation of the Wiener process, equal to sqrt(h) * diffusion
diff_mat     -- structure matrix for the diffusion coefficients, replaces the default uniform scaling
state_dim    -- keyword for parameter estimation, dimension of the dynamic state < dimension of full extended state
param_sample -- ParamSample dictionary for merging extended state with dx_params
ξ            -- random array size state_dim, can be defined in kwargs to provide a particular realization of the Wiener process


# dictionary for model parameters
ParamDict = Union{Dict{String, Array{Float64}}, Dict{String, Vector{Float64}}}

# dictionary containing key and index pairs to subset the state vector and merge with dx_params
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

### API for data assimilation solvers

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

analysis   -- string of the DA scheme string name, given to the transform sub-routine in methods
ens        -- ensemble matrix defined by the array with columns given by the replicates of the model state
obs        -- observation vector for the current analysis in ensemble_filter / array with columns given
              by the observation vectors for the ordered sequence of analysis times in the current 
							smoothing window
obs_cov    -- observation error covariance matrix, must be positive definite of type CovM
state_infl -- multiplicative covariance inflation factor for the state variable covariance matrix,
              set to one this is the standard Kalman-like update
kwargs     -- keyword arguments for parameter estimation or other functionality, including integration
              parameters for the state model in smoothing schemes
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

## Using experiments in dev package

In order to use the full experiment drivers, to both generate time series for the truth twin and for
running the twin experiments, one must have access to the sub-modules in the `experiments` directory.

### GenerateTimeSeries

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

### FilterExps

The `FilterExps.jl` sub-module configures twin experiments using stored time series data as generated above for 
efficiency when using the same base-line time series to generate possibly different experiment configurations.
Experiment configurations are generated by
```{julia}
filter_state(args::Tuple{String,String,Int64,Float64,Int64,Float64,Int64,Float64})
time_series, method, seed, obs_un, obs_dim, γ, N_ens, infl = args
```
where `time_series` specifies the path to the .jld2 truth twin, `method` specifies the filter scheme, `seed` specifies
the pseudo-random seed, `obs_un` specifies the observation error standard deviation, assuming a uniform scaling observation
error covariance, `obs_dim` specifies the dimension of the observation vector, `γ` specifies the level of the nonliearity
in the `alternating_obs_operator`, `N_ens` specifies the ensemble size and `infl` specifies the (static) multiplicative inflation.

Similar conventions follow for parameter estimation experiments
```{julia}
filter_param(args::Tuple{String,String,Int64,Float64,Int64,Float64,Float64,Float64,Int64,Float64,Float64})
time_series, method, seed, obs_un, obs_dim, γ, param_err, param_wlk, N_ens, state_infl, param_infl = args
```
with the exception of including the standard deviation, `param_err`, of the initial iid Gaussian draw of the parameter sample
centered at the true value, the standard deviation, `param_wlk`, of the random walk applied to the parameter sample
after each analysis and the multiplicative covariance inflation applied separately to the extended parameter
states alone `param_infl`.

The number of analyses in the twin experiment `nanl` is hard-coded currently in the function body of each experiment and
should be adjusted there.  Results are saved in the `data` directory according to the method used, assuming that the experiment
is run from the `experiments` directory or the `slurm_submit_scripts` directory.

### SmootherExps

The `SmootherExps.jl` sub-module configures twin experiments using stored time series data as generated above for 
efficiency when using the same base-line time series to generate possibly different experiment configurations.
Experiment configurations are generated by function calls as with the filter experiments, but with the additional
options of how the outer-loop is configured with a classic, single-iteration or the fully iterative Gauss-Newton style smoother.
The parameters `lag` and `shift` specify how the data assimilation windows are translated in over the observation
and analysis times.  The `mda` parameter is only applicable to the single-iteration and Gauss-Newton style smoothers,
utlizing sequential multiple data assimilation.  Note, the single-iteration and fully iterative Gauss-Newton style smoothers are
only defined for MDA compatible values of lag and shift where the lag is an integer multiple of the shift.

Currently debugged and validated smoother experiment configurations include
```
classic_state          -- classic EnKS style state estimation
classic_param          -- classic EnKS style state-parameter estimation
single_iteration_state -- single-iteration EnKS state estimation
single_iteration_param -- single-iteration EnKS state-parameter estimation
iterative_state        -- Gauss-Newton style state estimation
iterative_param        -- Gauss-Newton style state-parameter estimation
```
Other techniques are still in debugging and validation.  Each of these takes an analysis type as used in the
`transform` function in the `EnsembleKalmanSchemes.jl` sub-module, like the filter analyses in the filter experiments.

### SingleExperimentDriver / ParallelExperimentDriver

While the above filter experiments and smoother experiments configure twin experiments, run them and save the outputs,
the `SingleExperimentDriver.jl` and `ParallelExperimentDriver.jl` can be used as wrappers to run generic model settings for
debugging and validation, or to use built-in Julia parallelism to run a collection experiments over a parameter grid.
The `SingleExperimentDriver.jl` is primarily for debugging purposes with tools like `BenchmarkTools.jl` and `Debugger.jl`,
so that standard inputs can be run with the experiment called with macros.

The `ParallelExperimentDriver.jl` is a simple parallel implementation, though currently lacks a soft-fail when numerical
instability is encountered.  This means that if a single experiment configuration in the collection fails due to overflow, the entire
collection will cancel.  A fix for this is being explored, but the recommendation is to use the slurm submit
scripts below as templates for generating large parameter grid configurations and running them on servers.

### SlurmExperimentDrivers

These are a collection of templates for automatically generating an array of parameter tuples to pass to the experiment
functions as configurations.  This uses a simple looping strategy, while writing out the configurations to a .jld2 file
to be read by the parallel experiment driver within the `slurm_submit_scripts` directory.  The paralell submit script 
should be run within the `slurm_submit_scripts` directory to specify the correct paths to the time series data, the
experiment configuration data and to save to the correct output directory, specified by the method used.

### Processing experiment outputs

The `analysis` directory contains scripts for batch processing the outputs from experiments into time-averaged
RMSE and spread and arranging these outputs in an array for plotting.  This should be modified based on the
local paths to stored data.  This will try to load files based on parameter settings written in the name of
the output .jld2 file and if this is not available, this will store `Inf` values in the place of missing data.

### Validating results
Benchmark configurations for the above filtering and smoothing experiments are available in the open access article
[A fast, single-iteration ensemble Kalman smoother for sequential data assimilation](https://gmd.copernicus.org/preprints/gmd-2021-306/),
with details on the algorithm and parameter specifications discussed in the experiments section.  Performance of filtering and
smoothing schemes should be validated versus the numerical results presented there for root mean square error and ensemble spread.
Formal test cases for the package are currently in development.  The deterministic Runge-Kutta and Euler scheme for ODEs are
validated in the package tests, estimating the order of convergence with the least-squares log-10 line fit between step size
and discretization error.  Test cases for the stochastic integration schemes are in development, but numerical results with these
schemes can be validated versus the results in the open-access article 
[On the numerical integration of the Lorenz-96 model, with scalar additive noise, for benchmark twin experiments](https://gmd.copernicus.org/articles/13/1903/2020/).


## To do

  * Build additional tests for the library
  * Expand on the existing schemes and models

