# Workflow 

This package is based around file input and output, with experiment configurations defined
as function arguments using the
[NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple)
data type.  A basic workflow to run a data assimilation twin
experiment is to first generate a time series for observations using a choice of
tuneable parameters using the [GenerateTimeSeries](@ref) submodule.  Once the time
series data is generated from one of the benchmark models, one can use this data as a
truth twin to generate pseudo-observations. This time series can thus be re-used over
multiple configurations of filters and smoothers, holding the pseudo-data fixed while
varying other hyper-parameters.  Test cases in this package model this workflow,
to first generate test data and then to implement a particular experiment based
on a parameter configuration to exhibit known behavior of the estimator, typically in terms
of forecast and analysis root mean square error (RMSE).

Standard configurations of hyper-parameters for the truth twin and the data assimilation
method are included in the [SingleExperimentDriver](@ref) submodule, and constructors for
generating maps of parallel experiments over parameter grids are defined in the
[ParallelExperimentDriver](@ref) submodule.  It is assumed that one will
[Install a dev package](@ref) this package in order to define new parameter tuples
and constructors for parallel experiments in order to test the behavior of estimators
in new configurations.  It is also assumed that one will write new experiments using
the [FilterExps](@ref) and [SmootherExps](@ref) submodules as templates.
