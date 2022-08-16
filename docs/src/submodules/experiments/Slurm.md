# Slurm

### SlurmExperimentDrivers

These are a collection of templates for automatically generating an array of parameter tuples to pass to the experiment
functions as configurations.  This uses a simple looping strategy, while writing out the configurations to a .jld2 file
to be read by the parallel experiment driver within the `slurm_submit_scripts` directory.  The paralell submit script 
should be run within the `slurm_submit_scripts` directory to specify the correct paths to the time series data, the
experiment configuration data and to save to the correct output directory, specified by the method used.


