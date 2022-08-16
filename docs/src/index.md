# Description

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
in the pre-print
[A fast, single-iteration ensemble Kalman smoother for sequential data
assimilation](https://gmd.copernicus.org/preprints/gmd-2021-306/)
available currently in open review in Geoscientific Model Development.
