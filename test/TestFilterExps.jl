##############################################################################################
module TestFilterExps
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks
using DataAssimilationBenchmarks.SingleExperimentDriver
using DataAssimilationBenchmarks.FilterExps
using JLD2, Statistics
##############################################################################################
# run and analyze the ETKF for state estimation with the Lorenz-96 model

function run_ensemble_filter_state_L96()
    try
        ensemble_filter_state(exps["Filter"]["L96_ETKF_state_test"])
        true
    catch
        false
    end
end

function analyze_ensemble_filter_state_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/etkf/"
        data = load(path * "etkf_L96_state_seed_0000_diff_0.000_sysD_40_obsD_40" *
                    "_obsU_1.00_gamma_001.0_nanl_03500_tanl_0.05_h_0.05_nens_021" *
                    "_stateInfl_1.02.jld2")
        rmse = data["filt_rmse"]

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

function run_ensemble_filter_param_L96()
    try
        ensemble_filter_param(exps["Filter"]["L96_ETKF_param_test"])
        true
    catch
        false
    end
end

function analyze_ensemble_filter_param_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/etkf/"
        data = load(path * "etkf_L96_param_seed_0000_diff_0.000_sysD_41_stateD_40_obsD_40_" *
                    "obsU_1.00_gamma_001.0_paramE_0.10_paramW_0.0010_nanl_03500_tanl_0.05_" *
                    "h_0.05_nens_021_stateInfl_1.02_paramInfl_1.00.jld2")
        filt_rmse = data["filt_rmse"]
        para_rmse = data["param_rmse"]

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
function run_ensemble_filter_state_IEEE39bus()
    try
        ensemble_filter_state(exps["Filter"]["IEEE39bus_ETKF_state_test"])
        true
    catch
        false
    end
end

function analyze_ensemble_filter_state_IEEE39bus()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/etkf/"
        data = load(path * "etkf_IEEE39bus_state_seed_0000_diff_0.000_sysD_20_obsD_20_" *
                    "obsU_0.10_gamma_001.0_nanl_03500_tanl_0.01_h_0.01_nens_021_" *
                    "stateInfl_1.02.jld2")
        rmse = data["filt_rmse"]

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
