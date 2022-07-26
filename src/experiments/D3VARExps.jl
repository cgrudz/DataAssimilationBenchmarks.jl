##############################################################################################
module D3VARExps
##############################################################################################
# imports and exports
using Random, Distributions, LinearAlgebra, StatsBase
using ..DataAssimilationBenchmarks, ..ObsOperators, ..DeSolvers, ..XdVAR
##############################################################################################
# Main 3DVAR experiments
##############################################################################################

#=function D3_var_filter_analysis((time_series, method, seed, nanl, lag, shift, obs_un, obs_dim,
                        γ, N_ens, s_infl)::NamedTuple{
                        (:time_series,:method,:seed,:nanl,:lag,:shift,:obs_un,:obs_dim,
                        :γ,:N_ens,:s_infl),
                        <:Tuple{String,String,Int64,Int64,Int64,Int64,Float64,Int64,
                            Float64,Int64,Float64}})=#
function D3_var_filter_analysis(x::VecA(T)) where T <: Real
    
    # time the experiment
    t1 = time()

    # Define experiment parameters
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

    # Need to ask about this
    # define the initialization
    # observation noise
    v = rand(Normal(0, 1), 40)

    # define the observation range and truth reference solution
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
    

    for k in 1:f_steps
        # M(x^b)
        step_model!(x_b, 0.0, kwargs)
        # M(x^t)
        step_model!(x_t, 0.0, kwargs)
    end

    w = rand(MvNormal(zeros(40), I))
    obs = x_t + w

    state_cov = I
    obs_cov = I

    # generate cost function
    J = XdVAR.D3_var_cost(x, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    print("Cost Function Output: \n")
    display(J)
    # optimize cost function
    opt = XdVAR.D3_var_NewtonOp(x, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    print("Optimized Cost Function: \n")
    display(opt)
    # compare forecast and optimal state via RMSE
    rmse = sqrt(msd(x_b, opt))
    print("RMSE between Forecast and Optimal State: ")
    display(rmse)
    # output time
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
end

##############################################################################################
# end module

end