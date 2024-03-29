##############################################################################################
module SingleExperimentDriver
##############################################################################################
# imports and exports
using JLD2, HDF5
using ..DataAssimilationBenchmarks, ..FilterExps, ..SmootherExps, ..GenerateTimeSeries
export time_series_exps, enkf_exps, d3_var_exps, classic_enks_exps, sienks_exps, ienks_exps
##############################################################################################
# Utility methods and definitions
##############################################################################################

path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"

##############################################################################################
# Time series
##############################################################################################
"""
    time_series_exps = Dict{String, Any}

Pre-configured time series experiments for generating test data for twin experiments.
"""
time_series_exps = Dict{String, Any}(
    # Generates a short time series of the L96 model for testing
    "L96_deterministic_test" => (
      seed      = 0,
      h         = 0.05,
      state_dim = 40,
      tanl      = 0.05,
      nanl      = 5000,
      spin      = 1500,
      diffusion = 0.00,
      F         = 8.0,
     ),

    # Generates a short time series of the IEEE39bus model for testing
    "IEEE39bus_deterministic_test" => (
      seed      = 0,
      h         = 0.01,
      tanl      = 0.01,
      nanl      = 5000,
      spin      = 1500,
      diffusion = 0.0,
     ),
    )

##############################################################################################
# EnKF style experiments 
##############################################################################################
"""
    enkf_exps = Dict{String, Any}

Pre-configured EnKF style twin experiments for benchmark models. 
"""
enkf_exps = Dict{String, Any}(
    # Lorenz-96 ETKF state estimation standard configuration
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

    # Lorenz-96 3D-VAR state estimation standard configuration
    "L96_D3_var_state_test" => (
      time_series = path *
      "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_" *
      "spin_1500_h_0.050.jld2",
      bkg_cov     = "ID",
      seed        = 0,
      nanl        = 3500,
      obs_un      = 1.0,
      obs_dim     = 40,
      γ           = 1.00,
      s_infl      = 0.20,
     ),

    # Lorenz-96 ETKF joint state-parameter estimation standard configuration
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

    # IEEE39bus ETKF state estimation standard configuration
    "IEEE39bus_ETKF_state_test" => (
      time_series = path *
      "IEEE39bus_time_series_seed_0000_diff_0.000_tanl_0.01_nanl_05000_spin_1500_" *
      "h_0.010.jld2",
      method  = "etkf",
      seed    = 0,
      nanl    = 3500,
      obs_un  = 0.1,
      obs_dim = 20,
      γ       = 1.00,
      N_ens   = 21,
      s_infl  = 1.02
     ),
    )

##############################################################################################
# 3D-VAR style experiments
##############################################################################################
"""
    d3_var_exps = Dict{String, Any}

Pre-configured 3D-VAR style twin experiments for benchmark models. 
"""
d3_var_exps = Dict{String, Any}(
    # Lorenz-96 3D-VAR state estimation configuration
    "L96_D3_var_state_test" => (
      time_series = path *
      "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_05000_"*
      "spin_1500_h_0.050.jld2",
      bkg_cov     = "ID",
      seed        = 0,
      nanl        = 3500,
      obs_un      = 1.0,
      obs_dim     = 40,
      γ           = 1.00,
      s_infl      = 0.23,
     )
    )

##############################################################################################
# Classic smoothers
##############################################################################################
"""
    classic_enks_exps = Dict{String, Any}

Pre-configured classic EnKS style twin experiments for benchmark models. 
"""
classic_enks_exps = Dict{String, Any}(
    # Lorenz-96 ETKS state estimation standard configuration
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

    # Lorenz-96 ETKS joint state-parameter estimation standard configuration
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
    )


##############################################################################################
# SIEnKS style experiments 
##############################################################################################
"""
    sienks_exps = Dict{String, Any}

Pre-configured SIEnKS style twin experiments for benchmark models. 
"""
sienks_exps = Dict{String, Any}(
    # Lorenz-96 SIEnKS sda state estimation standard configuration
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
    # Lorenz-96 SIEnKS mda state estimation standard configuration
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

    # Lorenz-96 SIEnKS sda joint state-parameter estimation standard configuration
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

    # Lorenz-96 SIEnKS mda joint state-parameter estimation standard configuration
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
    )

##############################################################################################
# IEnKS style experiments 
##############################################################################################
"""
    ienks_exps = Dict{String, Any}

Pre-configured IEnKS style twin experiments for benchmark models. 
"""
ienks_exps = Dict{String, Any}(
    # Lorenz-96 IEnKS sda state estimation standard configuration
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

    # Lorenz-96 IEnKS mda state estimation standard configuration
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

    # Lorenz-96 IEnKS sda joint state-parameter estimation standard configuration
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

    # Lorenz-96 IEnKS mda joint state-parameter estimation standard configuration
    "L96_IEnKS_param_mda_test" => (
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
    )


##############################################################################################
# end module

end
