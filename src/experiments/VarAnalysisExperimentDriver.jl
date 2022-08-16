##############################################################################################
module VarAnalysisExperimentDriver
##############################################################################################
# imports and exports
using ..DataAssimilationBenchmarks
export D3_var_filter_tuning

##############################################################################################
# Utility methods and definitions
##############################################################################################

path = pkgdir(DataAssimilationBenchmarks) * "/src/analysis/var_exp/"

##############################################################################################
# Filters
##############################################################################################
"""
    args, exp = D3_var_filter_tuning()
Constucts a parameter map and experiment wrapper for sensitivity test of covariance tuning.
"""
function D3_var_filter_tuning()

    exp = DataAssimilationBenchmarks.D3VARExps.D3_var_filter_analysis
    function wrap_exp(arguments)
        try
            exp(arguments)
        catch
            print("Error on " * string(arguments) * "\n")
        end
    end

    # set filter parameters
    time_series = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2"
    γ = 1.0
    is_informed = true
    is_updated = true

    # define tuning range
    tuning_min = 0.01
    tuning_step = 0.001
    tuning_max = 0.03
    
    # load the experiments
    args = Vector{Any}()
    for t in tuning_min:tuning_step:tuning_max
        tmp = (
                time_series = time_series,
                γ = γ,
                is_informed = is_informed,
                tuning_factor = t,
                is_updated = is_updated
            )
        push!(args, tmp)
    end
    return args, wrap_exp
end


##############################################################################################
# end module

end