##############################################################################################
module D3VARExps
##############################################################################################
# imports and exports
using Random, Distributions, LinearAlgebra, StatsBase, Statistics
using JLD2, HDF5, Plots
using ..DataAssimilationBenchmarks, ..ObsOperators, ..DeSolvers, ..XdVAR
##############################################################################################
# Main 3DVAR experiments
##############################################################################################

function D3_var_filter_analysis_simple()
    
    # time the experiment
    t1 = time()

    # Define experiment parameters
    # number of cycles in experiment
    nanl = 40
    # load the timeseries and associated parameters
    # ts = load(time_series)::Dict{String,Any}
    # diffusion = ts["diffusion"]::Float64
    diffusion = 0.0
    # dx_params = ts["dx_params"]::ParamDict(Float64)
    # tanl = ts["tanl"]::Float64
    tanl = 0.05
    # model = ts["model"]::String
    γ = [8.0]

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # set the integration step size for the ensemble at 0.01 - we are assuming SDE
    h = 0.01

    # define derivative parameter
    dx_params = Dict{String, Vector{Float64}}("F" => [8.0])

    # define the dynamical model derivative for this experiment from the name
    # supplied in the time series - we are assuming Lorenz-96 model
    dx_dt = L96.dx_dt

    # define integration method
    step_model! = rk4_step!

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # set seed
    seed = 234
    Random.seed!(seed)

    # define the initialization
    # observation noise
    v = rand(Normal(0, 1), 40)

    # define the initial observation range and truth reference solution
    x_b = zeros(40)
    x_t = x_b + v
    
    # define kwargs for the analysis method
    # and the underlying dynamical model
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "dx_params" => dx_params,
                              "h" => h,
                              "diffusion" => diffusion,
                              "gamma" => γ,
                             )

    # create storage for the forecast and analysis statistics
    fore_rmse = Vector{Float64}(undef, nanl)
    filt_rmse = Vector{Float64}(undef, nanl)
    
    for i in 1:nanl
        #print("Iteration: ")
        #display(i)
        for j in 1:f_steps
            # M(x^b)
            step_model!(x_b, 0.0, kwargs)
            # M(x^t)
            step_model!(x_t, 0.0, kwargs)
        end

    # multivariate - rand(MvNormal(zeros(40), I))
    w = rand(Normal(0, 1), 40)
    obs = x_t + w

    state_cov = I
    obs_cov = I

    # generate initial forecast cost
    # J_i = XdVAR.D3_var_cost(x_b, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    # print("Cost Function Output: \n")
    # display(J)
    # optimized cost function input and value
    x_opt = XdVAR.D3_var_NewtonOp(x_b, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    # J_opt = XdVAR.D3_var_cost(x_opt, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    # print("Optimized Cost Function Output: \n")
    # display(J_opt)

    # compare model forecast and truth twin via RMSE
    rmse_forecast = sqrt(msd(x_b, x_t))
    #print("RMSE between Model Forecast and Truth Twin: ")
    #display(rmse_forecast)
    fore_rmse[i] = rmse_forecast

    # compare optimal forecast and truth twin via RMSE
    rmse_filter = sqrt(msd(x_opt, x_t))
    #print("RMSE between Optimal Forecast and Truth Twin: ")
    #display(rmse_filter)
    filt_rmse[i] = rmse_filter

    # reinitializing x_b and x_t for next cycle
     x_b = x_opt
    end

    data = Dict{String,Any}(
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "gamma" => γ,
                            "tanl" => tanl,
                            "nanl" => nanl,
                            "h" =>  h,
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse
                           )

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"
    name = "L96_3DVAR_time_series_seed_" * lpad(seed, 4, "0") *
           "_diff_" * rpad(diffusion, 5, "0") *
           #"_F_" * lpad(, 4, "0") *
           "_tanl_" * rpad(tanl, 4, "0") *
           "_nanl_" * lpad(nanl, 5, "0") *
           "_h_" * rpad(h, 5, "0") *
           ".jld2"

    save(path * name, data)
    # output time
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
    # make plot
    t = 1:nanl
    plot(t, fore_rmse, marker=(:circle,5), label = "Forecast", title="Root-Mean-Square Error vs. Time")
    plot!(t, filt_rmse, marker=(:circle,5), label = "Filter")
    xlabel!("Time [Cycles]")
    ylabel!("Root-Mean-Square Error [RMSE]")
end


##############################################################################################

function D3_var_filter_analysis((time_series, γ, is_informed, tuning_factor, is_updated)::NamedTuple{
                        (:time_series,:γ,:is_informed,:tuning_factor,:is_updated),<:Tuple{String,
                            Float64,Bool,Float64,Bool}})
    
    # time the experiment
    t1 = time()

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    dx_params = ts["dx_params"]::ParamDict(Float64)
    tanl = ts["tanl"]::Float64
    nanl = ts["nanl"]::Int64
    # set the integration step size for the ensemble 
    h = ts["h"]::Float64

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # define the dynamical model derivative for this experiment
    # we are assuming Lorenz-96 model
    dx_dt = L96.dx_dt

    # define integration method
    step_model! = rk4_step!

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # set seed
    seed = ts["seed"]::Int64
    Random.seed!(seed)

    # define the initialization
    o = ts["obs"]::Array{Float64, 2}
    obs_un = 1
    obs_cov = obs_un^2.0 * I
    # define state covaraince based on input
    if is_informed == true
        c = cov(o, dims = 2)
        state_cov = tuning_factor*Symmetric(c)
    else
        state_cov = tuning_factor*I
    end

    x_t = o[:,1]
    # observation noise
    v = rand(MvNormal(zeros(40), I))
    # define the initial observation range and truth reference solution
    x_b = x_t + v;
    
    # define kwargs for the analysis method
    # and the underlying dynamical model
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "dx_params" => dx_params,
                              "h" => h,
                              "diffusion" => diffusion,
                              "γ" => γ,
                              "gamma" => γ,
                              "obs_un" => obs_un,
                              "obs_cov" => obs_cov,
                              "state_cov" => state_cov
                             )
    
    # create storage for the forecast and analysis statistics
    fore_rmse = Vector{Float64}(undef, nanl)
    filt_rmse = Vector{Float64}(undef, nanl)
    
    for i in 1:(nanl-1)
        for j in 1:f_steps
            # M(x^b)
            step_model!(x_b, 0.0, kwargs)
        end

        w = rand(MvNormal(zeros(40), I))
        obs = o[:, i+1] + w
    
        # optimized cost function input and value 
        x_opt = XdVAR.D3_var_NewtonOp(x_b, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    
        x_t = o[:, i+1]
        # compare model forecast and filter via RMSE
        rmse_forecast = sqrt(msd(x_b, x_t))
        fore_rmse[i] = rmse_forecast
        rmse_filter = sqrt(msd(x_opt, x_t))
        filt_rmse[i] = rmse_filter

        # reinitializing x_b for next cycle if updated
        if is_updated == true
            x_b = x_opt
        end
    end

    data = Dict{String,Any}(
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "gamma" => γ,
                            "γ" => γ,
                            "tanl" => tanl,
                            "nanl" => nanl,
                            "h" =>  h,
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse
                           )

    if is_informed == true
        inf = "true"
    else
        inf = "false"
    end

    if is_updated == true
        upd = "true"
    else
        upd = "false"
    end

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/d3_var_exp/"
    name = "D3_var_filter_analysis_" * "L96_time_series_seed_" * lpad(seed, 4, "0") *
           "_gam_" * rpad(γ, 5, "0") *
           "_Informed_" * lpad(inf, 4, "0") *
           "_Updated_" * lpad(upd, 4, "0") *
           "_Tuned_" * rpad(tuning_factor, 5, "0") *
           ".jld2"

    save(path * name, data)
    # output time
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

    #= path = pkgdir(DataAssimilationBenchmarks) * "/plots/"
    name = "D3_var_filter_analysis_" * "L96_time_series_seed_" * lpad(seed, 4, "0") *
           "_gam_" * rpad(γ, 5, "0") *
           "_Informed_" * lpad(is_informed, 4, "0") *
           "_Updated_" * rpad(is_informed, 4, "0") *
           "_Tuned_" * lpad(tuning_factor, 5, "0")
    # make plot
    t = 1:nanl
    fore_rmse_ra = Vector{Float64}(undef, nanl)
    filt_rmse_ra = Vector{Float64}(undef, nanl)

    for i in 1:nanl
        fore_rmse_ra[i] = sum(fore_rmse[1:i])/i
        filt_rmse_ra[i] = sum(filt_rmse[1:i])/i
    end

    plot(t, fore_rmse_ra, label = "Forecast", title="Average Analysis RMSE vs. Time")
    plot!(t, filt_rmse_ra, label = "Filter")
    plot!([0, 5000], [1, 1])
    xlabel!("Time [Cycles]")
    ylabel!("Average Analysis RMSE")
    savefig(path * name)=#
end

##############################################################################################

function D3_var_filter_analysis_tune((time_series, γ, is_informed, tuning_min, tuning_step, tuning_max, is_updated)::NamedTuple{
    (:time_series,:γ,:is_informed,:tuning_min,:tuning_step,:tuning_max,:is_updated),<:Tuple{String,
        Float64,Bool,Float64,Float64,Float64,Bool}})

# time the experiment
t1 = time()

# load the timeseries and associated parameters
ts = load(time_series)::Dict{String,Any}
diffusion = ts["diffusion"]::Float64
dx_params = ts["dx_params"]::ParamDict(Float64)
tanl = ts["tanl"]::Float64
nanl = ts["nanl"]::Int64
# set the integration step size for the ensemble 
h = ts["h"]::Float64

# define the observation operator HARD-CODED in this line
H_obs = alternating_obs_operator

# define the dynamical model derivative for this experiment
# we are assuming Lorenz-96 model
dx_dt = L96.dx_dt

# define integration method
step_model! = rk4_step!

# number of discrete forecast steps
f_steps = convert(Int64, tanl / h)

# set seed
seed = ts["seed"]::Int64
Random.seed!(seed)

# define the initialization
o = ts["obs"]::Array{Float64, 2}
obs_un = 1
obs_cov = obs_un^2.0 * I
# define state covaraince based on input
if is_informed == true
    c = cov(o, dims = 2)
    state_cov = Symmetric(c)
else
    state_cov = I
end

# define kwargs for the analysis method
# and the underlying dynamical model
kwargs = Dict{String,Any}(
          "dx_dt" => dx_dt,
          "f_steps" => f_steps,
          "step_model" => step_model!,
          "dx_params" => dx_params,
          "h" => h,
          "diffusion" => diffusion,
          "γ" => γ,
          "gamma" => γ,
          "obs_un" => obs_un,
          "obs_cov" => obs_cov,
          "state_cov" => state_cov
         )

tuning_cycles = Int64((tuning_max - tuning_min)/tuning_step)
index_count = 1
# create storage for the forecast and analysis statistics
stab_fore_rmse = Vector{Float64}(undef, tuning_cycles)
stab_filt_rmse = Vector{Float64}(undef, tuning_cycles)

for t in tuning_min:tuning_step:tuning_max
    x_t = o[:,1]
    # observation noise
    v = rand(MvNormal(zeros(40), I))
    # define the initial background
    x_b = x_t + v;

    # create storage for the forecast and analysis statistics
    fore_rmse = Vector{Float64}(undef, nanl)
    filt_rmse = Vector{Float64}(undef, nanl)

    for i in 1:(nanl-1)
        for j in 1:f_steps
        # M(x^b)
        step_model!(x_b, 0.0, kwargs)
        end

        w = rand(MvNormal(zeros(40), I))
        obs = o[:, i+1] + w

        # optimized cost function input and value 
        x_opt = XdVAR.D3_var_NewtonOp(x_b, obs, x_b, t*state_cov, H_obs, obs_cov, kwargs)

        x_t = o[:, i+1]
        # compare model forecast and filter via RMSE
        rmse_forecast = sqrt(msd(x_b, x_t))
        fore_rmse[i] = rmse_forecast
        rmse_filter = sqrt(msd(x_opt, x_t))
        filt_rmse[i] = rmse_filter

        # reinitializing x_b for next cycle if updated
        if is_updated == true
            x_b = x_opt
        end
    end

    stab_fore_rmse[index_count] = sum(fore_rmse[1:nanl])/nanl
    stab_filt_rmse[index_count] = sum(filt_rmse[1:nanl])/nanl

    index_count = index_count + 1
end

data = Dict{String,Any}(
        "seed" => seed,
        "diffusion" => diffusion,
        "dx_params" => dx_params,
        "gamma" => γ,
        "γ" => γ,
        "tanl" => tanl,
        "nanl" => nanl,
        "h" =>  h,
        "stab_fore_rmse" => stab_fore_rmse,
        "stab_filt_rmse" => stab_filt_rmse
       )

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/d3_var_exp/"
    name = "D3_var_filter_analysis_" * "L96_time_series_seed_" * lpad(seed, 4, "0") *
            "_gam_" * rpad(γ, 5, "0") *
            "_Informed_" * lpad(is_informed, 4, "0") *
            "_Updated_" * rpad(is_informed, 4, "0") *
            "_TuneMin_" * lpad(tuning_min, 5, "0") *
            "_TuneStep_" * lpad(tuning_step, 5, "0") *
            "_TuneMax_" * lpad(tuning_max, 5, "0") *
            ".jld2"
   
    save(path * name, data)

    # output time
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

    # make plot
    path = pkgdir(DataAssimilationBenchmarks) * "/plots/"
    name = "D3_var_filter_analysis_" * "L96_time_series_seed_" * lpad(seed, 4, "0") *
            "_gam_" * rpad(γ, 5, "0") *
            "_Informed_" * lpad(is_informed, 4, "0") *
            "_Updated_" * rpad(is_informed, 4, "0") *
            "_TuneMin_" * lpad(tuning_min, 5, "0") *
            "_TuneStep_" * lpad(tuning_step, 5, "0") *
            "_TuneMax_" * lpad(tuning_max, 5, "0")

    t = tuning_min:tuning_step:tuning_max
    plot(t, stab_fore_rmse, label = "Forecast", title="Stabilized Analysis RMSE vs. Tuning Parameter")
    plot!(t, stab_filt_rmse, label = "Filter")
    plot!([0, 5000], [1, 1])
    xlabel!("Tuning Parameter")
    ylabel!("Stabilized Analysis RMSE")
    savefig(path * name)
end

##############################################################################################
# end module

end