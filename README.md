# DataAssimilationBenchmarks.jl

![DataAssimilationBenchmarks.jl logo](https://github.com/cgrudz/DataAssimilationBenchmarks.jl/blob/master/assets/dabenchmarks.png)

[![DOI](https://zenodo.org/badge/268903920.svg)](https://zenodo.org/badge/latestdoi/268903920)
[![status](https://joss.theoj.org/papers/478dcc0b1608d2a4d8c930edebb58736/status.svg)](https://joss.theoj.org/papers/478dcc0b1608d2a4d8c930edebb58736)
[![Total lines of code without comments](https://tokei.rs/b1/github/cgrudz/DataAssimilationBenchmarks.jl?category=code)](https://github.com/cgrudz/DataAssimilationBenchmarks.jl)
[![Build Status](https://app.travis-ci.com/cgrudz/DataAssimilationBenchmarks.jl.svg?branch=master)](https://app.travis-ci.com/cgrudz/DataAssimilationBenchmarks.jl)
[![codecov](https://codecov.io/gh/cgrudz/DataAssimilationBenchmarks.jl/branch/master/graph/badge.svg?token=3XLYTH8YSZ)](https://codecov.io/gh/cgrudz/DataAssimilationBenchmarks.jl)

## Welcome to DataAssimilationBenchmarks.jl!

### Description
This is my personal data assimilation benchmark research code with an emphasis on testing and validation
of ensemble-based filters and sequential smoothers in chaotic toy models.  The code is meant to be performant, 
in the sense that large hyper-parameter discretizations can be explored to determine structural sensitivity 
and reliability of results across different experimental regimes, with parallel implementations in Slurm.
This includes code for developing and testing data assimilation schemes in the 
[L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) currently, with further models in development.
This project supported the development of all numerical results and benchmark simulations considered in the pre-print
[A fast, single-iteration ensemble Kalman smoother for sequential data assimilation](https://gmd.copernicus.org/preprints/gmd-2021-306/)
available currently in open review in Geoscientific Model Development.

Lines of code counter (without comments or blank lines) courtesy of [Tokei](https://github.com/XAMPPRocky/tokei).
