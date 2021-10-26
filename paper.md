---
title: 'DataAssimilationBenchmarks.jl: a data assimilation research framework.'
tags:
  - Julia
  - Data Assimilation
  - Optimization
  - ensemble Kalman filter
  - ensemble Kalman smoother
	- ensemble-variational
authors:
  - name: Colin Grudzien
    orcid: 0000-0002-3084-3178
    affiliation: 1 
  - name: Sukhreen Sandhu 
    affiliation: 1
affiliations:
 - name: University of Nevada, Reno
   index: 1
date: 30 November 2021
bibliography: paper.bib
---

# Summary

Data assimilation (DA) refers to techniques used to combine the data from physics-based,
numerical models and real-world observations to produce an estimate for the state of a
time-evolving random process and the parameters that govern its evolution. Owing to their
history in numerical weather prediction, DA systems are designed to operate in an extremely
large dimension of model variables and observations, often with sequential-in-time observational
data. As a long-studied "big-data" problem, DA has benefited from the fusion of a variety
of techniques, including methods from Bayesian inference, dynamical systems, numerical analysis,
optimization, control theory and machine learning. DA techniques are widely used in many
areas of geosciences, neurosciences, biology, autonomous vehicle guidance and various
engineering applications requiring dynamic state estimation and control.

# Statement of need

While analytical proofs can be derived for classical methods such as the Kalman filter
in linear-Gaussian dynamics, most DA techniques are designed for estimation in nonlinear,
non-Gaussian models where no analytical solution may exist.  Similar to nonilnear optimization,
DA methods, therefore, must be studied with rigorous numerical simulation in standard test-cases
to demonstrate the effectiveness and computational performance of novel algorithms.  Pursuant
to proposing a novel DA method, one should likewise compare the performance of a proposed scheme
with other standard methods within the same class of estimators.

Standard libraries exist for full-scale DA system research and development, e.g.,
the Data Assimilation Research Testbed (DART)[@anderson2009data], but
there are fewer standard options for low-dimensional model demonstration and theoretical
research.  DataAssimilationBenchmarks.jl provides one framework for studying ensemble-based
filters and sequential smoothers that are commonly used in online, geoscientific prediction
settings.  Validated methods and methods in development focus on evaluating the performance
and the stuctural stability of techniques over wide ranges of hyper-parameters that are
commonly used to tune techniques in practice.  Specifically, this is designed to run naively
parallel experiment configurations over indepdent parameters such as ensemble size, static
covariance inflation, observation operator / network designs that affet the estimator
stability and performance.  Templates for running naively parallel experiments using Juila's
core parallelism, or using Slurm to load experiments in parallel in a queueing system are
provided.  This software framework was written to support the development and intercomparison
of the novel single-iteration ensemble Kalman smoother (SIEnKS) [@grudzien2021fast], including
the inter-comparison with other popular ensemble-variational, maximum-a-posteriori estimators
following an ensemble Kalman filter (EnKF)-based analysis.

## Comparison with similar projects

Similar projects to DataAssimlationBenchmarks.jl include the DAPPER Python library
[@patrick_n_raanes_2018_2029296], DataAssim.jl used by [@vetra2018state4e], and
EnsembleKalmanProcesses.jl [@enkprocesses] of the Climate Modeling Alliance.  These alternatives
are differentiated primarily in that:
<ul>
	<li>DAPPER is a Python-based library which is well-established, and includes many of the same
	estimators and models. However, DAPPER is notably slower due to its dependence on the Python
	language for its core numerical techniques.  This can make the wide hyper-parameter search
	intended above computationally challenging.</li>
	<li>DataAssim.jl is another established Julia library, but notably lacks an implementation
	of ensemble-variational techniques which were the focus of the initial development of
	DataAssimilationBenchmkarks.jl.  For this reason, this package was not selected for the 
	development and intercomparison of the SIEnKS, though this package does have implementations
	of a variety of standard stochastic filtering schemes.</li>
	<li>EnsembleKalmanProcesses.jl is another established Julia library, but notably lacks
	traditional DA approaches such as the classic, perturbed observation EnKF and the classic
	ETKF.  For this reason, this package was not selected for the development and intercomparison
	of the SIEnKS.</li>
</ul>

## Validated methods currently in use

<table>
<tr>
	<th>Estimator / implemented techniques</th>
	<th>Tuned multiplicative inflation</th>
	<th>Adaptive inflation, finite-size formalism (perfect model dual / primal)</th>
	<th>Linesearch</th>
	<th>Multiple data assimilation (general shift and lag)</th>
</tr>
<tr>
  <td>EnKF, perturbed obs.</td><td>X</td><td>X</td><td>NA</td><td>NA</td>
</tr>
<tr>
  <td>ETKF</td><td>X</td><td>X</td><td>NA</td><td>NA</td>
</tr>
<tr>
  <td>MLEF, transform / bundle variants</td><td>X</td><td>X</td><td>X</td><td>NA</td>
</tr>
<tr>
  <td>EnKS, perturbed obs.</td><td>X</td><td>X</td><td>NA</td><td>NA</td>
</tr>
<tr>
  <td>ETKS</td><td>X</td><td>X</td><td>NA</td><td>NA</td>
</tr>
<tr>
  <td>MLES, transform / bundle variants</td><td>X</td><td>X</td><td>X</td><td>NA</td>
</tr>
<tr>
  <td>SIEnKS, perturbed obs / ETKF / MLEF variants</td><td>X</td><td>X</td><td>X</td><td>X</td>
</tr>
<tr>
  <td>Gauss-Newton IEnKS, transform / bundle variants</td><td>X</td><td>X</td><td></td><td>X</td>
</tr>
</table>

The future development of the DataAssimilationBenchmarks.jl package is inteneded to expand upon
the existing, ensemble-variational filters and sequential smoothers for robust intercomparison of
novel schemes and the further development of the SIEnKS scheme.  Likewise, novel mechanistic models
for the DA system are currently in development. Currently, this supports state and joint
state-parameter estimation in the L96-s model [@grudzien2020numerical] in both ordinary
and stochastic differential equation formulations.  Likewise, this supports a variety of observation
operator configurations in the L96-s model, as outlined in [@grudzien2021fast].

# Installation

The main module DataAssimilationBenchmarks.jl is a wrapper module including the core numerical solvers 
for ordinary and stochastic differential equations, solvers for DA routines and the core 
process model code for running twin experiments with benchmark models. These methods can be run 
stand-alone in other programs by calling these functions from the DeSolvers, EnsembleKalmanSchemes and 
L96 sub-modules from this library. Future solvers and models will be added as sub-modules in the methods
and models directories respectively. 

In order to get the full functionality of this package one needs to 
need to install the dev version.  This provides the access to edit all of the outer-loop routines for 
setting up twin experiments. These routines are defined in the modules in the "experiments" directory.
The "slurm_submit_scripts" directory includes routines for parallel submission of experiments in Slurm.
Data processing scripts and visualization scripts (written in Python with Matplotlib and Seaborn) are 
included in the "analysis" directory.

## Installing a dev package from the Julia General registries 

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

# Acknowledgements
Colin Grudzien wrote all core numerical code in the DataAssmilationBenchmarks.jl package.  Sukhreen
Sandhu supported development of the package by building and validating code test cases.  This work
was supported by the University of Nevada, Reno, Office of Undergraduate Research's
Pack Research Experience Program (PREP) which supported Sukhreen Sandhu as a research assistant.
This work benefitted from the DAPPER library which was referenced at times for the development
of DA schemes.

# References
