# ParallelExperimentDriver 

In order to perform sensitivity testing and estimator tuning, many different parameter
combinations may need to be evaluated for each experiment defined in the submodules 
[GenerateTimeSeries](@ref), [FilterExps](@ref) and [SmootherExps](@ref).  These experiments
are designed so that these hyper-parameter searches can be implemented with naive parallelism,
using [parallel maps](https://en.wikipedia.org/wiki/MapReduce) and Julia's
native [Distributed](https://docs.julialang.org/en/v1/stdlib/Distributed/) computing module.

This module defines argumentless functions to construct an array with each array entry given
by a [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple), defining
a particular hyper-parameter configuration.  These functions also define a soft-fail method
for evaluating experiments, with example syntax as
```{julia}
args, wrap_exp = method()
```
where the `wrap_exp` follows a convention of

```{julia}
function wrap_exp(arguments)
    try
        exp(arguments)
    catch
        print("Error on " * string(arguments) * "\n")
    end
end
```
with `exp` being imported from one of the experiment modules above.

This soft-fail wrapper provides that if a single experiment configuration in the parameter
array fails due to, e.g., numerical overflow, the remaining configurations will continue
their own course unaffected.

## Example usage

An example of how one can use the ParallelExperimentDriver framework to run a sensitivity
test is as follows. We use a sensitivity test on the ensemble size for several
variants of the EnKF using adaptive inflation.  The following function, defined in
ParallelExperimentDriver.jl module, will construct all of input data for the truth twin
and a collection of NamedTuples that define individual experiments:

```{julia}
path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"

function ensemble_filter_adaptive_inflation()

    exp = DataAssimilationBenchmarks.FilterExps.ensemble_filter_state
    function wrap_exp(arguments)
        try
            exp(arguments)
        catch
            print("Error on " * string(arguments) * "\n")
        end
    end

    # set time series parameters
    seed      = 123
    h         = 0.05
    state_dim = 40
    tanl      = 0.05
    nanl      = 6500
    spin      = 1500
    diffusion = 0.00
    F         = 8.0

    # generate truth twin time series
    GenerateTimeSeries.L96_time_series(
                                       (
                                         seed      = seed,
                                         h         = h,
                                         state_dim = state_dim,
                                         tanl      = tanl,
                                         nanl      = nanl,
                                         spin      = spin,
                                         diffusion = diffusion,
                                         F         = F,
                                        )
                                      )

    # define load path to time series
    time_series = path * "L96_time_series_seed_" * lpad(seed, 4, "0") *
                         "_dim_" * lpad(state_dim, 2, "0") *
                         "_diff_" * rpad(diffusion, 5, "0") *
                         "_F_" * lpad(F, 4, "0") *
                         "_tanl_" * rpad(tanl, 4, "0") *
                         "_nanl_" * lpad(nanl, 5, "0") *
                         "_spin_" * lpad(spin, 4, "0") *
                         "_h_" * rpad(h, 5, "0") *
                         ".jld2"

    # define ranges for filter parameters
    methods = ["enkf-n-primal", "enkf-n-primal-ls", "enkf-n-dual"]
    seed = 1234
    obs_un = 1.0
    obs_dim = 40
    N_enss = 15:3:42
    s_infls = [1.0]
    nanl = 4000
    γ = 1.0
    
    # load the experiments
    args = Vector{Any}()
    for method in methods
        for N_ens in N_enss
            for s_infl in s_infls
                tmp = (
                       time_series = time_series,
                       method = method,
                       seed = seed,
                       nanl = nanl,
                       obs_un = obs_un,
                       obs_dim = obs_dim,
                       γ = γ,
                       N_ens = N_ens,
                       s_infl = s_infl
                      )
                push!(args, tmp)
            end
        end
    end
    return args, wrap_exp
end

```

With a constructor as above, one can define a script as follows to run the sensitivity test:

```{julia}
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

args, wrap_exp = config()
num_exps = length(args)

print("Configuration ready\n")
print("\n")
print("Running " * string(num_exps) * " configurations on " * string(nworkers()) *
      " total workers\n")
print("Begin pmap\n")
pmap(wrap_exp, args)
print("Experiments completed, verify outputs in the appropriate directory under:\n")
print(pkgdir(DataAssimilationBenchmarks) * "/src/data\n")

##############################################################################################
# end module

end
```

Running the script using
```
julia -p N run_sensitivity_test.jl
```
will map the evaluation of all parameter configurations to parallel workers where `N`
is the number of workers, to be defined based on the available resources on the user system.
User-defined sensitivity tests can be generated by modifying the above script according
to new constructors defined within the ParallelExperimentDriver module.

## Experiment groups
```@autodocs
Modules = [DataAssimilationBenchmarks.ParallelExperimentDriver]
```
