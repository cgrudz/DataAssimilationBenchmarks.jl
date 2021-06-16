# DataAssimilationBenchmarks

[![Build Status](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/workflows/CI/badge.svg)](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/actions)
[![Coverage](https://codecov.io/gh/cgrudz/DataAssimilationBenchmarks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/cgrudz/DataAssimilationBenchmarks.jl)
[![Coverage](https://coveralls.io/repos/github/cgrudz/DataAssimilationBenchmarks.jl/badge.svg?branch=master)](https://coveralls.io/github/cgrudz/DataAssimilationBenchmarks.jl?branch=master)

## Description
This is my personal data asimilation benchmark research code with an emphasis on testing and validation of ensemble-based filters and smoothers in chaotic toy models.  This includes code for developing and testing data assimilation schemes in the L96-s model currently, with further models in development.

## Structure
The directory is structured as follows:
  * models - contains code for the dynamic model equations.
  * methods - contains DA methods and general numerical methodological code. 
  * experiments - contains the scripts that set up twin experiments
  * data - input / output directory for the inputs to and ouptuts from experiments.

## To do
  * Begin general packaging and documentation.
