##############################################################################################
module TestIterativeSmootherExps
##############################################################################################
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks.SmootherExps
using JLD2, Statistics

##############################################################################################
##############################################################################################
# run and analyze the IEnKS for state estimation with the Lorenz-96 model
# arguments are
# time_series, method, seed, nanl, lag, shift, mda, obs_un, obs_dim, γ, N_ens, infl = args

function run_sda_smoother_state_L96()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        L96_test_ts = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0" *
                      "_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2"
        
        args = (path * L96_test_ts, "ienks-transform", 0, 3500, 10, 1, false,
                1.0, 40, 1.00, 21, 1.02)
        iterative_state(args)
        true
    catch
        false
    end
end

function analyze_sda_smoother_state_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = joinpath(@__DIR__, "../src/data/ienks-transform/") 
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

function run_mda_smoother_state_L96()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        L96_test_ts = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0" *
                      "_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2"
        
        args = (path * L96_test_ts, "ienks-transform", 0, 3500, 10, 1, true,
                1.0, 40, 1.00, 21, 1.02)
        iterative_state(args)
        true
    catch
        false
    end
end

function analyze_mda_smoother_state_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = joinpath(@__DIR__, "../src/data/ienks-transform/") 
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
# time_series, method, seed, nanl, lag, shift, mda, obs_un, obs_dim, γ,
# param_err, param_wlk, N_ens, state_infl, param_infl = args

function run_sda_smoother_param_L96()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        L96_test_ts = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0" *
                      "_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2"
        
        args = (path * L96_test_ts, "ienks-transform", 0, 3500, 10, 1, false, 1.0, 40, 1.0,
                0.10, 0.0010, 21, 1.02, 1.0)
        iterative_param(args)
        true
    catch
        false
    end
end

function analyze_sda_smoother_param_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = joinpath(@__DIR__, "../src/data/ienks-transform/") 
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

function run_mda_smoother_param_L96()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        L96_test_ts = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0" *
                      "_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2"
        
        args = (path * L96_test_ts, "ienks-transform", 0, 3500, 10, 1, true, 1.0, 40, 1.0,
                0.10, 0.0010, 21, 1.02, 1.0)
        iterative_param(args)
        true
    catch
        false
    end
end

function analyze_mda_smoother_param_L96()
    try
        # test if the filter RMSE for standard simulation falls below adequate threshold
        path = joinpath(@__DIR__, "../src/data/ienks-transform/") 
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
