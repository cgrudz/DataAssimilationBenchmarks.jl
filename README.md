# da_benchmark

## Description
This is my personal data asimilation benchmark research code.  This includes code for developing and testing data assimilation schemes in the L96-s model and the nonlinear swing equations.

## Structure
The directory is structured as follows:
  * models - contains code for the dynamic model equations.
  * methods - contains DA methods and general numerical methodological code. 
  * experiments - contains the scripts that set up twin experiments
  * data - dummy output directory for consistency with the server

## To do
	* Write the DEnKF update? This is also a simple extension of the single-iteration formalism
	* Decide if adaptive inflation in each scheme with MDA will be used, estimating power of the distribution
	* get final publication version of the state estimation code prepared
	* begin work on the stochastic model / parameter estimation / adaptive inflation schemes
