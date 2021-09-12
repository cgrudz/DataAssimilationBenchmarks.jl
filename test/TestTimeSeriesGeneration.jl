#######################################################################################################################
module TestTimeSeriesGeneration
#######################################################################################################################
# imports and exports
using DeSolvers
using L96
using JLD
export kwargs, time_series
#######################################################################################################################
function time_series()
    # initial conditions and arguments
    state_dim = 40

    seed = 123
    x = zeros(state_dim)

    # total number of analyses
    nanl = 1000

    # diffusion parameter
    diffusion = 0.0

    # step size
    h = 0.01
    # forced parameter
    F = 8.0
    # time between analyses
    tanl = 0.5

    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => diffusion,
        "dx_params" => [F],
        "dx_dt" => L96.dx_dt,
        )

    step_model! = DeSolvers.em_step!

    f_steps = convert(Int64, tanl/h)

    obs = Array{Float64}(undef, state_dim, nanl)

    for i in range(1, stop=nanl)
        for j in range(1, stop=f_steps)
            em_step!(x, 0.0, kwargs)
        end
        obs[:, i] = x
    end

    data = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "F" => F,
            "tanl" => tanl,
            "nanl" => nanl,
            "state_dim" => state_dim,
            "obs" => obs
           )

    path = "../data/time_series/"
    fname = "time_series_data_seed_" * string(seed) * ".jld"
    save(path * fname, data)

end

end
