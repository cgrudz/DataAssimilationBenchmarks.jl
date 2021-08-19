#######################################################################################################################
module TestTimeSeriesGeneration
#######################################################################################################################
# imports and exports
using DeSolvers

#######################################################################################################################

function l96(x, f)
    x_m_2  = cat(x[end-1:end], x[1:end-2])
    x_m_1 = cat(x[end:end], x[1:end-1])
    x_p_1 = cat(x[2:end], x[1:1], dims = 2)

    dxdt = (x_p_1-x_m_2).*x_m_1 - x + f

end
#one step forward

f = 8.0
spin = 100
h = 0.001
nanl = 1000
sys_dim = 40
diffusion = 0.1
tanl = 0.1
seed = 0

#fore_steps = int(tanl/h)
fore_steps = (tanl/h)

#np.random.seed(seed)
using Random
Random.seed!(seed)


#xt = np.ones(sys_dim)
xt = ones(sys_dim)

#for i in range(int(spin / h)):
    #xt = l96_em_sde(xt, h, [f, diffusion])

#tobs = np.zeros ([sys_dim, nanl])
tobs = zeros(sys_dim, nanl)
#for i in range(nanl)
for i in range(0, stop = nanl)
    #for j in range(fore_steps)
    for j in range(0, stop = fore_steps)
        xt = l96_em_sde(xt, h, [f, diffusion])
        tobs[1:i] = xt
    end
end

params = [spin, h, diffusion, f, seed]
#time_series = ["tobs": tobs, "params": params]
time_series = ["tobs"::tobs, "params"::params]
fname = "time_series_data_seed_" * string(seed) * ".txt"
#f = open(fname, "wb")
f = open(fname, "w")
#pickle.dump (time_series, f)
dumpval = dump(fname)
f.close()
end
