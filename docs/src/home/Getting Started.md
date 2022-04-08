# Getting Started

## Installation

The main module DataAssimilationBenchmarks.jl wraps sub-modules including the core numerical solvers
for ordinary and stochastic differential equations, solvers for data assimilation routines and the core
process model code for running twin experiments with benchmark models. These methods can be run
stand-alone in other programs by calling these functions from the DeSolvers, EnsembleKalmanSchemes,
L96 and IEEE39bus sub-modules from this library. Future solvers and models will be added as sub-modules
in the methods and models directories respectively.

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


### Repository structure
The repository is structured as follows:

```@raw html
<ul>
  <li> src - contains the main parent module</li>
  <ul>
		<li> models - contains code for defining the dynamic model equations in twin experiments.</li>
		<li> methods - contains DA solvers and general numerical routines for running twin experiments.</li>
		<li> experiments - contains the outer-loop scripts that set up twin experiments.</li>
		<li> data - this is an input / output directory for the inputs to and ouptuts from experiments.</li>
		<li> analysis - contains auxilliary scripts for batch processing experiment results and for plotting in Python.</li>
	</ul>
  <li> scratch - this is a storage directory for backups.</li>
  <li> test - contains test cases for the package.</li>
	<li> docs - contains the documenter files.</li>
</ul>
```

