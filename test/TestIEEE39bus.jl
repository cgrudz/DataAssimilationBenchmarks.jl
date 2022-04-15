##############################################################################################
module TestIEEE39bus
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks
using JLD, Statistics
##############################################################################################
"""
    test_synchrony() 

This function tests to see if the swing equation model without noise reaches the synchronous
steady state for the system, by evaluating the standard deviation of the state components
after the spin up period.
"""
function test_synchrony()
    try
        # load the observations
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"
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
