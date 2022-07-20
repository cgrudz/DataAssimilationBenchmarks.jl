# Getting Started

## Installation
The main module DataAssimilationBenchmarks.jl declares global types and type constructors.
These conventions are utilized in sub-modules that implement the core numerical solvers
for ordinary and stochastic differential equations, solvers for data assimilation routines
and the core process model code for running twin experiments with benchmark models, collected
in the `methods` and `models` sub-directories.  Experiments define routines for driving
standard benchmark case studies with
[NamedTuples](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple)
as arguments to these methods defining the associated experimental hyper-parameters.

In order to get the full functionality of this package you will need to install the dev
version. This provides access to create new experiments and to define performance benchmarks
for these experiments

### Installing a dev package from the Julia General registries
In order to install the dev version to your Julia environment, you can use the following
commands in the REPL
```{julia}
pkg> dev DataAssimilationBenchmarks
```
The installed version will be included in your
```
~/.julia/dev/
```
on Linux and the analogous directory with respect Windows and Mac systems.
Alternatively, you can install this from the main Github branch directly as follows:
```{julia}
pkg> dev https://github.com/cgrudz/DataAssimilationBenchmarks.jl
```

### Repository structure
The repository is structured as follows:
```@raw html
<ul>
  <li><code>src</code> - contains the main parent module</li>
  <ul>
		<li><code>models</code> - contains code for defining the dynamic model equations in twin
		experiments.</li>
		<li><code>methods</code> - contains DA solvers and general numerical routines for running
		twin experiments.</li>
		<li><code>experiments</code> - contains the outer-loop scripts that set up twin
		experiments.</li>
		<li><code>data</code> - this is an input / output directory for the inputs to and
		ouptuts from experiments.</li>
		<li><code>analysis</code> - contains auxilliary scripts for batch processing experiment
		results and for plotting (currently in Python).</li>
	</ul>
  <li><code>scratch</code> - this is a storage directory for backups.</li>
  <li><code>test</code> - contains test cases for the package.</li>
	<li><code>docs</code> - contains the documenter files.</li>
</ul>
```
