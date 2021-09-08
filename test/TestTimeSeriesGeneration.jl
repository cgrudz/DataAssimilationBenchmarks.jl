#######################################################################################################################
module TestTimeSeriesGeneration
#######################################################################################################################
# imports and exports
using DataAssimilationBenchmarks.DeSolvers
using DataAssimilationBenchmarks.L96
using JLD
using Random

#######################################################################################################################
function time_series()
    # initial conditions and arguments
    state_dim = 40


dx_dt = L96.dx_dt
step_model! = DeSolvers.em_step!
#one step forward

F = 8.0
spin = 100
h = 0.001
nanl = 1000
sys_dim = 40
diffusion = 0.1
tanl = 0.1
seed = 0

#fore_steps = int(tanl/h)
fore_steps = convert(Int64, tanl/h)

#np.random.seed(seed)
Random.seed!(seed)

kwargs = Dict{String, Any}(
          "h" => h,
          "diffusion" => diffusion,
          "dx_params" => [F],
          "dx_dt" => L96.dx_dt,
         )

#xt = np.ones(sys_dim)
xt = ones(sys_dim)

#for i in range(int(spin / h)):
    #xt = l96_em_sde(xt, h, [f, diffusion])

#tobs = np.zeros ([sys_dim, nanl])
tobs = Array{Float64}(undef,sys_dim, nanl)
#for i in range(nanl)
for i in 1:nanl
    #for j in range(fore_steps)
    for j in 1:fore_steps
        step_model!(xt, 0.0, kwargs)
    end
    tobs[:,i] = xt
end


fname = "time_series_data_seed_" * string(seed) * ".jld"

data = Dict{String, Any}(
    "h" => h,
    "diffusion" => diffusion,
    "F" => F,
    "tanl" => tanl,
    "nanl" => nanl,
    "sys_dim" => sys_dim,
    "tobs" => tobs
    )
path = "../data/time_series/"
save(path * fname, data)

end
