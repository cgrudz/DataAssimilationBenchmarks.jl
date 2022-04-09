# Description

This is my personal data assimilation benchmark research code with an emphasis on testing and validation
of ensemble-based filters and sequential smoothers in chaotic toy models.  The code is meant to be performant,
in the sense that large hyper-parameter discretizations can be explored to determine structural sensitivity
and reliability of results across different experimental regimes, with parallel implementations in Slurm.
This includes code for developing and testing data assimilation schemes in the
[L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) currently, with further models in development.
This project supported the development of all numerical results and benchmark simulations considered in the pre-print
[A fast, single-iteration ensemble Kalman smoother for sequential data assimilation](https://gmd.copernicus.org/preprints/gmd-2021-306/)
available currently in open review in Geoscientific Model Development.

