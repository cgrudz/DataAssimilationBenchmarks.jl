# DataAssimilationBenchmarks

## Description
This is my personal data asimilation benchmark research code with an emphasis on testing and validation of ensemble-based filters and smoothers in chaotic toy models.  The code is meant to be performant, in the sense that large hyper-parameter discretizations can be explored to determine structural sensitivity and reliability of results across different experimental regimes, with parallel implementations in Slurm.  This includes code for developing and testing data assimilation schemes in the [L96-s model](https://gmd.copernicus.org/articles/13/1903/2020/) currently, with further models in development.

## Structure
The directory is structured as follows:
  * models - contains code for the dynamic model equations.
  * methods - contains DA methods and general numerical methodological code. 
  * experiments - contains the scripts that set up twin experiments
  * data - input / output directory for the inputs to and ouptuts from experiments.

## To do
  * Begin general packaging and documentation.
