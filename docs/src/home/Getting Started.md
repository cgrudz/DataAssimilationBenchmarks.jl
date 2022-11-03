# Getting Started

## Installation

The main module DataAssimilationBenchmarks.jl declares global types and type constructors.
These conventions are utilized in sub-modules that implement the core numerical solvers
for ordinary and stochastic differential equations, solvers for data assimilation routines,
and the core process model code for running twin experiments with benchmark models, collected
in the `methods` and `models` sub-directories.  Experiments define routines for driving
standard benchmark case studies with
[NamedTuples](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple)
as arguments to these methods defining the associated experimental hyper-parameters.

This parent module only serves to support the overhead of type declarations used thoughout
the package and the functionality of the methods standalone is extremely limited.
In order to get the full functionality of this package __you will need to install the dev
version__. This provides access to the source code needed to create new experiments and
to define performance benchmarks for these experiments.

### Install a dev package

There are two ways to install a dev package of the repository.
In either case, the installed version will be included in your
```
~/.julia/dev/
```
on Linux and the analogous directory with respect Windows and Mac systems.

#### Install the tagged stable version

To install the last tagged official release, you can use the following
commands in the REPL
```{julia}
pkg> dev DataAssimilationBenchmarks
```
This version in the Julia General Registries will be the latest official release.
However, this official release tends to lag behind the current version.

#### Install the up-to-date version

You can install the latest version from the main Github branch directly as follows:
```{julia}
pkg> dev https://github.com/cgrudz/DataAssimilationBenchmarks.jl
```
The master branch synchronizes with the up-to-date documentation and commits to the master
branch are considered tested but not necessarily stable.  As this package functions as a
__research framework__, the master branch is in continuous development.  If your use case is
performing research of DA methods with this package, it is recommended to install and keep
up-to-date with the current version of the master branch.

### Repository structure

The repository is structured as follows:
```@raw html
<ul>
  <li><code>src</code> - contains the main parent module</li>
  <ul>
		<li><code>models</code> - contains code for defining the state and observation model equation for twin
		experiments</li>
		<li><code>methods</code> - contains DA solvers and general numerical routines for running
		twin experiments</li>
		<li><code>experiments</code> - contains the outer-loop scripts that set up twin
		experiments, and constructors for generating parameter grids</li>
		<li><code>data</code> - this is an input / output directory for the inputs to and
		ouptuts from experiments</li>
		<li><code>analysis</code> - contains auxilliary scripts for batch processing experiment
		results and for plotting (currently in Python, not fully integrated).</li>
	</ul>
  <li><code>test</code> - contains test cases for the package.</li>
	<li><code>docs</code> - contains the documenter files.</li>
</ul>
```
