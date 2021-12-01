##############################################################################################
module TestIEEE39bus
##############################################################################################
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks.DeSolvers, DataAssimilationBenchmarks.L96
using JLD, Statistics

##############################################################################################
##############################################################################################
# Test to see if the model without noise reaches the synchronous steady state

function test_synchrony()
    try
        # load the observations
        path = joinpath(@__DIR__, "../src/data/time_series/")
        obs = load(path * "IEEE39bus_time_series_seed_0000_diff_0.000_tanl_0.01" *
                   "_nanl_05000_spin_1500_h_0.010.jld2")
        obs = obs["obs"][:, 3001:end]

        # take the standard deviation of the model state after warm up
        sd = std(obs, dims=2)
        if sum(sd .< 0.01) == 20
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
