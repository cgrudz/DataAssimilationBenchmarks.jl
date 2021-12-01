##############################################################################################
module TestFilterExps
##############################################################################################
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks.FilterExps
using JLD2, Statistics

##############################################################################################
##############################################################################################
# run and analyze the ETKF for state estimation with the Lorenz-96 model

function run_filter_state_L96()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        L96_test_ts = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0" *
                      "_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2"
        
        args = (path * L96_test_ts, "etkf", 0, 3500, 1.0, 40, 1.00, 21, 1.02)
        filter_state(args)
        true
    catch
        false
    end
end

function analyze_filter_state_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = joinpath(@__DIR__, "../src/data/etkf/") 
        rmse = load(path * "etkf_L96_state_seed_0000_diff_0.000_sysD_40_obsD_40" * 
                    "_obsU_1.00_gamma_001.0_nanl_03500_tanl_0.05_h_0.05_nens_021" *
                    "_stateInfl_1.02.jld2")
        rmse = rmse["filt_rmse"]

        # note, we use a small burn-in to reach more regular cycles
        if mean(rmse[501:end]) < 0.2
            true
        else
            false
        end
    catch
        false
    end
end


##############################################################################################
# run and analyze the ETKF for joint state-parameter estimation with the Lorenz-96 model

function run_filter_param_L96()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        L96_test_ts = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0" *
                      "_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2"
        
        args = (path * L96_test_ts, "etkf", 0, 3500, 1.0, 40, 1.0, 0.10, 0.0010, 21, 1.02, 1.0)
        filter_param(args)
        true
    catch
        false
    end
end

function analyze_filter_param_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = joinpath(@__DIR__, "../src/data/etkf/") 
        rmse = load(path * "etkf_L96_param_seed_0000_diff_0.000_sysD_41_stateD_40_obsD_40_" *
                    "obsU_1.00_gamma_001.0_paramE_0.10_paramW_0.0010_nanl_03500_tanl_0.05_" *
                    "h_0.05_nens_021_stateInfl_1.02_paramInfl_1.00.jld2")
        filt_rmse = rmse["filt_rmse"]
        para_rmse = rmse["param_rmse"]

        # note, we use a small burn-in to reach more regular cycles
        if (mean(filt_rmse[501:end]) < 0.2) && (mean(para_rmse[501:end]) < 0.01)
            true
        else
            false
        end
    catch
        false
    end
end


##############################################################################################
# run and analyzed the ETKF for state estimateion with the IEEE39bus model

# static version for test cases
function run_filter_state_IEEE39bus()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        IEEE39bus_test_ts = "IEEE39bus_time_series_seed_0000_diff_0.000_tanl" * 
                            "_0.01_nanl_05000_spin_1500_h_0.010.jld2"
        args = (path * IEEE39bus_test_ts, "etkf", 0, 3500, 0.1, 20, 1.00, 21, 1.02)
        filter_state(args)
        true
    catch
        false
    end
end

function analyze_filter_state_IEEE39bus()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = joinpath(@__DIR__, "../src/data/etkf/") 
        rmse = load(path * "etkf_IEEE39bus_state_seed_0000_diff_0.000_sysD_20_obsD_20_" * 
                    "obsU_0.10_gamma_001.0_nanl_03500_tanl_0.01_h_0.01_nens_021_" *
                    "stateInfl_1.02.jld2")
        rmse = rmse["filt_rmse"]

        # note, we use a small burn-in to reach more regular cycles
        if mean(rmse[501:end]) < 0.02
            true
        else
            false
        end
    catch
        false
    end
end


##############################################################################################
# end module

end
