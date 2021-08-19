# DataAssimilationBenchmarks.jl

```
  _____        _                         _           _ _       _   _              
 |  __ \      | |          /\           (_)         (_) |     | | (_)             
 | |  | | __ _| |_ __ _   /  \   ___ ___ _ _ __ ___  _| | __ _| |_ _  ___  _ __   
 | |  | |/ _` | __/ _` | / /\ \ / __/ __| | '_ ` _ \| | |/ _` | __| |/ _ \| '_ \  
 | |__| | (_| | || (_| |/ ____ \\__ \__ \ | | | | | | | | (_| | |_| | (_) | | | | 
 |_____/ \__,_|\__\__,_/_/    \_\___/___/_|_| |_| |_|_|_|\__,_|\__|_|\___/|_| |_| 

  ____                  _                          _          _ _
 |  _ \                | |                        | |        (_) |                
 | |_) | ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____   _| |                
 |  _ < / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __| | | |                
 | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \_| | |                
 |____/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___(_) |_|                
                                                            _/ |                  
                                                           |__/                   
```
## Welcome to DataAssimilationBenchmarks.jl!

### Description
This is my personal data asimilation benchmark research code with an emphasis on testing and validation
of ensemble-based filters and sequential smoothers in chaotic toy models.  The code is meant to be performant, 
in the sense that large hyper-parameter discretizations can be explored to determine structural sensitivity 
and reliability of results across different experimental regimes, with parallel implementations in Slurm.
This includes code for developing and testing data assimilation schemes in the 
[L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) currently, with further models in development.

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
  <td>SIETKS</td><td>X</td><td>X</td><td></td><td>X</td><td></td><td>X</td>
</tr>
<tr>
  <td>Gauss-Newton IEnKS, transform / bundle variants</td><td>X</td><td>X</td><td></td><td></td><td></td><td>X</td>
</tr>
</table>

### Structure
The directory is structured as follows:
  * src - contains the main wrapper module
  * models - contains code for defining the dynamic model equations in twin experiments.
  * methods - contains DA solvers and general numerical routines for running twin experiments.
  * experiments - contains the outer-loop scripts that set up twin experiments.
  * data - this is an input / output directory for the inputs to and ouptuts from experiments.
  * scratch - this is a storage directory for backups
  * analysis - contains auxilliary scripts for batch processing experiment results and for plotting in Python
  * test - contains test cases for the package, currently under development

## Installation

The main package DataAssimilationBenchmarks.jl is a wrapper library including the core numerical solvers 
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


### Installing a dev package from Github

In order to install the dev version to your Julia environment, you can use the following commands in the REPL

```{julia}
(@v1.6) pkg> dev https://github.com/cgrudz/DataAssimilationBenchmarks.jl
```

The installed version will be included in your
```
~/.julia/dev/
```
on Linux and the analogous directory with respect Windows and Mac systems.

### Installing a dev package from the Julia General registries 

Currently DataAssimilationBenchmarks.jl is still pending integration into the Julia General registries.  When this is available,
instructions will be posted here.

## Using solvers in DataAssimilationBenchmarks

The methods directory currently contains two types of solvers, differential equation solvers in the DeSolvers.jl sub-module,
and ensemble-Kalman-filter-based data assimilation routines in the EnsembleKalmanSchemes.jl sub-module.

### API for differential equation solvers

Three general schemes are developed for ordinary and stochastic differential equations, the four-stage Runge-Kutta, second order Taylor,
and the Euler-(Maruyama) schemes.  Because the second order Taylor-Stratonovich scheme relies specifically on the structure of the
Lorenz-96 model with additive noise, this is included separately in the `models/L96.jl` sub-module.

General schemes such as the four-stage Runge-Kutta and the Euler-(Maruyama) schemes are built to take in arguments of the form

```{julia}
(x::T, t::Float64, kwargs::Dict{String,Any}) where {T <: VecA}
VecA = Union{Vector{Float64}, SubArray{Float64, 1}}

x          -- array or sub-array of a single state possibly including parameter values
t          -- time point
kwargs     -- this should include dx_dt, the paramters for the dx_dt and optional arguments
dx_dt      -- time derivative function with arguments x and dx_params
dx_params  -- tuple of parameters necessary to resolve dx_dt, not including parameters in the extended state vector 
h          -- numerical discretization step size
diffusion  -- tunes the standard deviation of the Wiener process, equal to sqrt(h) * diffusion
state_dim  -- keyword for parameter estimation, dimension of the dynamic state < dimension of full extended state
ξ          -- random array size state_dim, can be defined in kwargs to provide a particular realization of the Wiener process

```
with reduced options for the less-commonly used deterministic second order Taylor and first order Euler(-Maruyama) schemes.

The time steppers over-write the value of `x` in place as a vector or a view of an array for efficient ensemble integration.
We follow the convention in data assimilation of the extended state formalism for parameter estimation where the parameter sample
should be included as trailing state variables in the columns of the ensemble array.  If

```{julia}
true == haskey(kwargs, "state_dim")
```
the `state_dim` parameter will specify the dimension of the dynamical states and creat a view of the vector `x` including all entries
up to this index.  The remaining entries in the vector `x` will be passed to the `dx_dt` function in a vector including
`dx_params` in the leading entries and the extended state entries trailing.  The parameter sample values will remain unchanged
by the time stepper when the dynamical state entries in `x` are over-written in place.

Setting `diffusion>0` above introduces additive noise to the dynamical system.  The main `rk4_step!` has convergence on order 4.0
when diffusion is equal to zero, and both strong and weak convergence on order 1.0 when stochasticity is reduced.  Nonetheless,
this is the recommended out-of-the-box solver for any generic DA simulation for the robust performance over Euler-(Maruyama).
When specifically generating the truth-twin for the Lorenz96 model with additive noise, this should be performed with the
`l96s_tay2_step!` in the `models/L96.jl` sub-module while the ensemble should be generated with the `rk4_step!`.
See the benchmarks on the [L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) for a full discussion of
statistically robust model configurations.


### Currently debugged and validated methods

Write instructions here

## Using experiments in dev package

Write instructions here

### GenerateTimeSeries

Write instructions here

### FilterExps

Write instructions here

### SmootherExps

Write instructions here

### SingleExperimentDriver / ParallelExperimentDriver

Write instructions here

### SlurmExperimentDrivers

Write instructions here


## To do
  * Write installation instructions
	* Fill out more documentation on the API
	* Build tests for the library

