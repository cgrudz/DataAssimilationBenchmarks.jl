##############################################################################################
module SingleExperimentDriver
##############################################################################################
# imports and exports
using JLD2, HDF5
using ..DataAssimilationBenchmarks, ..FilterExps, ..SmootherExps, ..GenerateTimeSeries
export exps
##############################################################################################
# The following are standard inputs for experiments, written as named tuples and stored in a
# dictionary.  These standard inputs are used in the package for for debugging, testing, 
# benchmarking and profiling. Parallel submission scripts are used for performance on servers.
##############################################################################################
path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"

exps = Dict{String, Any}(
        "Generate_time_series" => Dict{String, Any}(
          "L96_deterministic_test" => (
            seed      = 0,
            state_dim = 40,
            tanl      = 0.05,
            nanl      = 5000,
            spin      = 1500,
            diffusion = 0.00,
            F         = 8.0,
           ),
          "IEEE39bus_deterministic_test" => (
            seed      = 0,
            tanl      = 0.01,
            nanl      = 5000,
            spin      = 1500,
            diffusion = 0.0,
           ),
          ),
        "Filter" => Dict{String, Any}(
          "L96_ETKF_state_test" => (
            time_series = path * 
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
            "spin_1500_h_0.050.jld2",
            method      = "etkf", 
            seed        = 0,
            nanl        = 3500,
            obs_un      = 1.0,
            obs_dim     = 40, 
            γ           = 1.00,
            N_ens       = 21,
            s_infl      = 1.02,
           ),
          "L96_ETKF_param_test" => (
            time_series = path * 
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
            "spin_1500_h_0.050.jld2",
            method  = "etkf",
            seed    = 0,
            nanl    = 3500,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.0,
            p_err   = 0.10,
            p_wlk   = 0.0010,
            N_ens   = 21,
            s_infl  = 1.02,
            p_infl  = 1.0,
           ),
          "IEEE39bus_ETKF_state_test" => (
            time_series = path *
            "IEEE39bus_time_series_seed_0000_diff_0.000_tanl_0.01_nanl_05000_spin_1500_" *
            "h_0.010.jld2",
            method="etkf",
            seed=0,
            nanl=3500,
            obs_un=0.1,
            obs_dim=20,
            γ=1.00,
            N_ens=21,
            s_infl=1.02
           ),
          ),
        "Classic_smoother" => Dict{String, Any}(
          "L96_ETKS_state_test" => (
            time_series = path *
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
            "spin_1500_h_0.050.jld2",
            method  = "etks",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.00,
            N_ens   = 21,
            s_infl  = 1.02,
           ),
          "L96_ETKS_param_test" => (
            time_series = path *
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
            "spin_1500_h_0.050.jld2",
            method  = "etks",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.0,
            p_err   = 0.10,
            p_wlk   = 0.0010,
            N_ens   = 21,
            s_infl  = 1.02,
            p_infl  = 1.0,
           ),
          ),
        "Single_iteration_smoother" => Dict{String, Any}(
          "L96_ETKS_state_sda_test" => (
            time_series = path * 
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
            "spin_1500_h_0.050.jld2",
            method  = "etks",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            mda     = false,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.00,
            N_ens   = 21,
            s_infl  = 1.02,
          ),
          "L96_ETKS_state_mda_test" => (
            time_series = path * 
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
            "spin_1500_h_0.050.jld2",
            method  = "etks",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            mda     = true,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.00,
            N_ens   = 21,
            s_infl  = 1.02,
           ),
          "L96_ETKS_param_sda_test" => (
            time_series = path *
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
            "spin_1500_h_0.050.jld2",
            method  = "etks",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            mda     = false,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.00,
            p_err   = 0.10,
            p_wlk   = 0.0010,
            N_ens   = 21,
            s_infl  = 1.02,
            p_infl  = 1.0,
           ),
          "L96_ETKS_param_mda_test" => (
            time_series = path * "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_" * 
            "tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2",
            method  = "etks",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            mda     = true,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.0,
            p_err   = 0.10,
            p_wlk   = 0.0010,
            N_ens   = 21,
            s_infl  = 1.02,
            p_infl  = 1.0,
           ),
          ),
        "Iterative_smoother" => Dict{String, Any}(
          "L96_IEnKS_state_sda_test" => (
            time_series = path *
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
            "spin_1500_h_0.050.jld2",
            method  = "ienks-transform",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            mda     = false,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.00,
            N_ens   = 21,
            s_infl  = 1.02,
           ),
          "L96_IEnKS_state_mda_test" => (
            time_series =  path *
            "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" * 
            "spin_1500_h_0.050.jld2",
            method  = "ienks-transform",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            mda     = true,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.00,
            N_ens   = 21,
            s_infl  = 1.02,
           ),
          "L96_IEnKS_param_sda_test" => (
            time_series = path * "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0" *
            "_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2",
            method  = "ienks-transform",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            mda     = false,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.00,
            p_err   = 0.10,
            p_wlk   = 0.0010,
            N_ens   = 21,
            s_infl  = 1.02,
            p_infl  = 1.0,
           ),
          "L96_IEnKS_param_sda_test" => (
            time_series = path * "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0" *
            "_tanl_0.05_nanl_05000_spin_1500_h_0.050.jld2",
            method  = "ienks-transform",
            seed    = 0,
            nanl    = 3500,
            lag     = 10,
            shift   = 1,
            mda     = true,
            obs_un  = 1.0,
            obs_dim = 40,
            γ       = 1.00,
            p_err   = 0.10,
            p_wlk   = 0.0010,
            N_ens   = 21,
            s_infl  = 1.02,
            p_infl  = 1.0,
           ),
          ),
         )


##############################################################################################
# end module

end
