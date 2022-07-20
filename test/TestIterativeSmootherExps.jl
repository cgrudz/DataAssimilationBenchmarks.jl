##############################################################################################
module TestIterativeSmootherExps
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks
using DataAssimilationBenchmarks.SmootherExps
using DataAssimilationBenchmarks.SingleExperimentDriver
using JLD2, Statistics
##############################################################################################
# run and analyze the IEnKS for state estimation with the Lorenz-96 model

function run_sda_ensemble_smoother_state_L96()
    try
        iterative_ensemble_state(exps["Iterative_smoother"]["L96_IEnKS_state_sda_test"])
        true
    catch
        false
    end
end

function analyze_sda_ensemble_smoother_state_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/ienks-transform/"
        data = load(path * "ienks-transform_L96_state_seed_0000_diff_0.000_sysD_40_obsD_40_" *
                    "obsU_1.00_gamma_001.0_nanl_03500_tanl_0.05_h_0.05_lag_010_shift_001_" *
                    "mda_false_nens_021_stateInfl_1.02.jld2")
        filt_rmse = data["filt_rmse"]
        post_rmse = data["post_rmse"]

        # note, we use a small burn-in to reach more regular cycles
        if mean(filt_rmse[501:end]) < 0.2 && mean(post_rmse[501:end]) < 0.10
            true
        else
            false
        end
    catch
        false
    end
end

function run_mda_ensemble_smoother_state_L96()
    try
        iterative_ensemble_state(exps["Iterative_smoother"]["L96_IEnKS_state_mda_test"])
        true
    catch
        false
    end
end

function analyze_mda_ensemble_smoother_state_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/ienks-transform/"
        data = load(path * "ienks-transform_L96_state_seed_0000_diff_0.000_sysD_40_obsD_40_" *
                    "obsU_1.00_gamma_001.0_nanl_03500_tanl_0.05_h_0.05_lag_010_shift_001_" *
                    "mda_true_nens_021_stateInfl_1.02.jld2")
        filt_rmse = data["filt_rmse"]
        post_rmse = data["post_rmse"]

        # note, we use a small burn-in to reach more regular cycles
        if mean(filt_rmse[501:end]) < 0.2 && mean(post_rmse[501:end]) < 0.10
            true
        else
            false
        end
    catch
        false
    end
end


##############################################################################################
# run and analyze the IEnKS for joint state-parameter estimation with the Lorenz-96 model

function run_sda_ensemble_smoother_param_L96()
    try
        iterative_ensemble_param(exps["Iterative_smoother"]["L96_IEnKS_param_sda_test"])
        true
    catch
        false
    end
end

function analyze_sda_ensemble_smoother_param_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/ienks-transform/"
        data = load(path * "ienks-transform_L96_param_seed_0000_diff_0.000_sysD_41_obsD_40_" *
                    "obsU_1.00_gamma_001.0_paramE_0.10_paramW_0.0010_nanl_03500_tanl_0.05_" *
                    "h_0.05_lag_010_shift_001_mda_false_nens_021_stateInfl_1.02_" *
                    "paramInfl_1.00.jld2")
        filt_rmse = data["filt_rmse"]
        post_rmse = data["post_rmse"]
        para_rmse = data["param_rmse"]

        # note, we use a small burn-in to reach more regular cycles
        if (mean(filt_rmse[501:end]) < 0.2) && (mean(post_rmse[501:end]) < 0.10) &&
            (mean(para_rmse[501:end]) < 0.01)
            true
        else
            false
        end
    catch
        false
    end
end

function run_mda_ensemble_smoother_param_L96()
    try
        iterative_ensemble_param(exps["Iterative_smoother"]["L96_IEnKS_param_mda_test"])
        true
    catch
        false
    end
end

function analyze_mda_ensemble_smoother_param_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/ienks-transform/"
        data = load(path * "ienks-transform_L96_param_seed_0000_diff_0.000_sysD_41_obsD_40_" *
                    "obsU_1.00_gamma_001.0_paramE_0.10_paramW_0.0010_nanl_03500_tanl_0.05_" *
                    "h_0.05_lag_010_shift_001_mda_true_nens_021_stateInfl_1.02_" *
                    "paramInfl_1.00.jld2")
        filt_rmse = data["filt_rmse"]
        post_rmse = data["post_rmse"]
        para_rmse = data["param_rmse"]

        # note, we use a small burn-in to reach more regular cycles
        if (mean(filt_rmse[501:end]) < 0.2) && (mean(post_rmse[501:end]) < 0.10) &&
            (mean(para_rmse[501:end]) < 0.01)
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
