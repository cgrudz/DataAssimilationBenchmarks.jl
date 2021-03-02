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
  * Write EnKF-N, this will be a simple extension of the hybrid approach to the ETKS
		* This needs to be benchmarked versus the tuned inflation to see the overal performance
	* Write the DEnKF update? This is also a simple extension of the hybrid formalism
	* Write the EnKS-N formalism for SDA
	* Determine the role of adaptive inflation in each scheme with MDA
