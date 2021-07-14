# DataAssimilationBenchmarks

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
## Welcome to DataAssimilationBenchmarks!

### Description
This is my personal data asimilation benchmark research code with an emphasis on testing and validation
of ensemble-based filters and smoothers in chaotic toy models.  The code is meant to be performant, in the
sense that large hyper-parameter discretizations can be explored to determine structural sensitivity and 
reliability of results across different experimental regimes, with parallel implementations in Slurm.
This includes code for developing and testing data assimilation schemes in the 
[L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) currently, with further models in development.

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

Write instructions here

### Installing a dev package from the Julia General registries 

Write instructions here

## Using solvers in DataAssimilationBenchmarks

Write instructions here

### General API for solvers

Write instructions here

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

