#######################################################################################################################
module TestTimeSeriesGeneration
#######################################################################################################################
# imports and exports
using DataAssimilationBenchmarks.DeSolvers
using DataAssimilationBenchmarks.L96
using JLD
using Random
#######################################################################################################################
# Test generation of the L96 model time series
function testL96()
# define the model and the solver
    dx_dt = L96.dx_dt
    step_model! = DeSolvers.em_step!

# set model and experimental parameters
    F = 8.0
    h = 0.001
    nanl = 1000
    sys_dim = 40
    diffusion = 0.1
    tanl = 0.01
    seed = 0
    Random.seed!(seed)

# define the dx_params dict
    dx_params = Dict{String, Array{Float64}}("F" => [F])
    fore_steps = convert(Int64, tanl/h)

# set the kwargs for the integration scheme
    kwargs = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "dx_params" => dx_params,
            "dx_dt" => L96.dx_dt,
            )

# set arbitrary initial condition
    xt = ones(sys_dim)

# pre-allocate storage for the time series observations
    tobs = Array{Float64}(undef,sys_dim, nanl)

# loop the experiment, taking observations at time length tanl
    for i in 1:nanl
        for j in 1:fore_steps
            step_model!(xt, 0.0, kwargs)
        end
        tobs[:,i] = xt
    end

# define the file name for the experiment output
# dynamically based on experiment parameters
    fname = "time_series_data_seed_" * lpad(seed, 4, "0") *
            "_dim_" * lpad(sys_dim, 2, "0") *
            "_diff_" * rpad(diffusion, 5, "0") *
            "_F_" * lpad(F, 4, "0") *
            "_tanl_" * rpad(tanl, 4, "0") *
            "_nanl_" * lpad(nanl, 5, "0") *
            "_h_" * rpad(h, 5, "0") *
            ".jld"

# define the experimental data in a dictionary to write with JLD
    data = Dict{String, Any}(
        "h" => h,
        "diffusion" => diffusion,
        "F" => F,
        "tanl" => tanl,
        "nanl"  => nanl,
        "sys_dim" => sys_dim,
        "tobs" => tobs
        )
        path = "../data/time_series/"

# test to see if the data can be written to standard output directory
    function write_file()
        try
            save(path * fname, data)
            did_write = true
        catch
        # if not, set test case false
            did_write = false
        end
    end

# test to see if the data can be read from standard output directory
    function load_file()
        try
            tmp = load(path * fname)
            did_read = true
        catch
        # if not, set test case false
            did_read = false
        end
    end

write_file()
load_file()
end

#######################################################################################################################

end
