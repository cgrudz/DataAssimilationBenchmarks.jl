##############################################################################################
module run_sensitivity_test 
##############################################################################################
# imports and exports
using Distributed
@everywhere using DataAssimilationBenchmarks

##############################################################################################

config = ParallelExperimentDriver.ensemble_filter_adaptive_inflation

print("Generating experiment configurations from " * string(config) * "\n")
print("Generate truth twin\n")

args, exp = config()
num_exps = length(args)

print("Configuration ready\n")
print("\n")
print("Running " * string(num_exps) * " configurations on " * string(nworkers()) *
      " total workers\n")
print("Begin pmap\n")
pmap(exp, args)
print("Experiments completed, verify outputs in the appropriate directory under:\n")
print(pkgdir(DataAssimilationBenchmarks) * "/src/data\n")

##############################################################################################
# end module

end
