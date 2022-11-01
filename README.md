# DataAssimilationBenchmarks.jl

![DataAssimilationBenchmarks.jl logo](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/blob/master/assets/dabenchmarks.png)

[![docs-dev-img](https://img.shields.io/badge/docs-dev-purple.svg)](https://cgrudz.github.io/DataAssimilationBenchmarks.jl/dev)
[![docs-stable-img](https://img.shields.io/badge/docs-stable-blue.svg)](https://cgrudz.github.io/DataAssimilationBenchmarks.jl/stable)
[![status](https://joss.theoj.org/papers/478dcc0b1608d2a4d8c930edebb58736/status.svg)](https://joss.theoj.org/papers/478dcc0b1608d2a4d8c930edebb58736)
[![Total lines of code without comments](https://tokei.rs/b1/github/cgrudz/DataAssimilationBenchmarks.jl?category=code)](https://github.com/cgrudz/DataAssimilationBenchmarks.jl)
[![Build Status](https://app.travis-ci.com/cgrudz/DataAssimilationBenchmarks.jl.svg?branch=master)](https://app.travis-ci.com/cgrudz/DataAssimilationBenchmarks.jl)
[![codecov](https://codecov.io/gh/cgrudz/DataAssimilationBenchmarks.jl/branch/master/graph/badge.svg?token=3XLYTH8YSZ)](https://codecov.io/gh/cgrudz/DataAssimilationBenchmarks.jl)

Lines of code counter (without comments or blank lines) courtesy of [Tokei](https://github.com/XAMPPRocky/tokei).

## Welcome to DataAssimilationBenchmarks.jl!

### Description

This is a data assimilation research code base with an emphasis on prototyping, testing and
validating sequential filters and smoothers in toy model twin experiments.
This code is meant to be performant in the sense that large hyper-parameter discretizations
can be explored to determine hyper-parameter sensitivity and reliability of results across
different experimental regimes, with parallel implementations in native Julia distributed
computing.

This package currently includes code for developing and testing data assimilation schemes in
the [L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) and the IEEE 39 bus test
case in the form of the [effective network
model](https://iopscience.iop.org/article/10.1088/1367-2630/17/1/015012)
model equations. New toy models and data assimilation schemes are in continuous development
in the development branch.  Currently validated techniques are available in the master
branch.

This package supported the development of all numerical results and benchmark simulations
in the manuscript 
[A fast, single-iteration ensemble Kalman smoother for sequential data
assimilation](https://gmd.copernicus.org/articles/15/7641/2022/gmd-15-7641-2022.html).

### Documentation

Please see the up-to-date documentation in the [master branch](https://cgrudz.github.io/DataAssimilationBenchmarks.jl/dev/)
or the [stable documentation](https://cgrudz.github.io/DataAssimilationBenchmarks.jl/dev/) for the last tagged version in
the [Julia General Registries](https://github.com/JuliaRegistries/General).
