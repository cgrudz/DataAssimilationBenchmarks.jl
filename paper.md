---
title: 'DataAssimilationBenchmarks.jl: a data assimilation research framework.'
tags:
  - Julia
  - Data Assimilation
	- Bayesian Inference
  - Optimization
  - Machine learning
authors:
  - name: Colin Grudzien
    orcid: 0000-0002-3084-3178
    affiliation: 1,2
  - name: Charlotte Merchant
    affiliation: 1,3 
  - name: Sukhreen Sandhu 
    affiliation: 4
affiliations:
 - name: CW3E - Scripps Institution of Oceanography, University of California, San Diego
   index: 1
 - name: Department of Mathematics and Statistics, University of Nevada, Reno
   index: 2
 - name: Department of Computer Science, Princeton University 
   index: 3
 - name: Department of Computer Science and Engineering, University of Nevada, Reno
   index: 4
date: 30 September 2022
bibliography: paper.bib
---

# Summary

Data assimilation (DA) refers to techniques used to combine the data from physics-based,
numerical models and real-world observations to produce an estimate for the state of a
time-evolving random process and the parameters that govern its evolution [@asch2016data]. 
Owing to their history in numerical weather prediction, full-scale DA systems are designed
to operate in an extremely large dimension of model variables and observations, often with
sequential-in-time observational data [@carrassi2018data]. As a long-studied "big-data"
problem, DA has benefited from the fusion of a variety of techniques, including methods
from Bayesian inference, dynamical systems, numerical analysis, optimization, control
theory and machine learning. DA techniques are widely used in many
areas of geosciences, neurosciences, biology, autonomous vehicle guidance and various
engineering applications requiring dynamic state estimation and control.

The purpose of this package is to provide a research framework for the theoretical
development and empirical validation of novel data assimilation techniques.
While analytical proofs can be derived for classical methods such as the Kalman filter
in linear-Gaussian dynamics [@jazwinski2007stochastic], most currently developed DA
techniques are designed for estimation in nonlinear, non-Gaussian models where no
analytical solution may exist.  DA methods,
therefore, must be studied with rigorous numerical simulation in standard test-cases
to demonstrate the effectiveness and computational performance of novel algorithms.
Pursuant to proposing a novel DA method, one should likewise compare the performance
of a proposed scheme with other standard methods within the same class of estimators.

This package implements a variety of standard data assimilation algorithms,
including some of the widely used performance modifications that are used in
practice to tune these estimators. This software framework was written originally
to support the development and intercomparison of methods studied in [@grudzien2021fast].
Details of the studied ensemble DA schemes, including pseudo-code detailing
their implementation, and DA experiment benchmark configurations, can be found in
the above principal reference.  Additional details on numerical integration schemes
utilized in this package can be found in the secondary reference [@grudzien2020numerical].

# Statement of need

Standard libraries exist for full-scale DA system research and development, e.g.,
the Data Assimilation Research Testbed (DART) [@anderson2009data], but
there are fewer standard options for theoretical research and algorithm development in
simple test systems. Many basic research frameworks, furthermore, do not include
standard operational techniques developed from classical variational methods,
due to the difficulty in constructing tangent linear and adjoint codes [@kalnay20074denkf].
DataAssimilationBenchmarks.jl provides one framework for studying squential filters
and smoothers that are commonly used in online, geoscientific prediction settings,
including both ensemble methods and variational schemes, with hybrid method planned for
future development.

## Comparison with similar projects

Similar projects to DataAssimilationBenchmarks.jl include the DAPPER Python library
[@dapper], DataAssim.jl used by [@vetra2018state], and
EnsembleKalmanProcesses.jl [@enkprocesses] of the Climate Modeling Alliance.
These alternatives are differentiated primarily in that:

  * DAPPER is a Python-based library which is well-established, and includes many of the same
	estimators and models. However, numerical simulations in Python run notably slower than
	simulations in Julia when numerical routines cannot be vectorized in Numpy
	[@bezanson2017julia]. Particularly, this can make the wide hyper-parameter search
	intended above computationally challenging without utilizing additional packages such
	as Numba [@lam2015numba] for code acceleration.
	
  * DataAssim.jl is another established Julia library, but notably lacks an implementation
	of variational and ensemble-variational techniques.
	
  * EnsembleKalmanProcesses.jl is another established Julia library, but notably lacks
	traditional geoscientific DA approaches such as 3D-VAR and the ETKF/S.

## Future development 

The future development of the DataAssimilationBenchmarks.jl package is intended to expand
upon the existing, variational and ensemble-variational filters and sequential smoothers for
robust intercomparison of novel schemes.  Additional process models and observation models 
for the DA system are also in development.

# Acknowledgements

Colin Grudzien developed the numerical code for the package's Julia type optimizations for
numerical schemes and automatic differentiation of code, the
ensemble-based estimation schemes, the observation models, the Lorenz-96 model, the IEEE 39
Bus test case model and the numerical integration schemes for ordinary and stochastic
differential equations.  Charlotte Merchant developed the numerical code for implementing
variational data assimilation in the Lorenz-96 model and related experiments. Sukhreen
Sandhu supported the development of the package structure and organization.
All authors supported the development of the package by implementing test cases.
This work was supported by the University of Nevada, Reno, Office of Undergraduate Research's
Pack Research Experience Program which supported Sukhreen Sandhu as a research assistant.
This work was supported by the Center for Western Weather and Water Extremes internship
program which supported Charlotte Merchant as a research assistant.
This work benefited from the DAPPER library which was referenced at times for the development
of DA schemes.  The authors would like to thank the handling editor Bita Hasheminezhad,
and the two named referees Lukas Riedel and Tangi Migot for their comments, suggestions
and valuable advice which strongly improved the quality of the paper and the software.

# References
