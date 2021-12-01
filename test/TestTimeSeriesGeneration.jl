##############################################################################################
module TestTimeSeriesGeneration
##############################################################################################
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks.GenerateTimeSeries
using JLD, Random

##############################################################################################
##############################################################################################
# Test generation and loading of the L96 model time series in default localtion

function testGenL96()
    try
        args = (0, 40, 0.05, 5000, 1500, 0.00, 8.0)
        L96_time_series(args)
        true
    catch
        false
    end
end

function testLoadL96()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        load(path * "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05" *
             "_nanl_05000_spin_1500_h_0.010.jld2")
        true
    catch
        false
    end
end


##############################################################################################
# Test generation and loading of the IEEE39 bus model time series in the default location

function testGenIEEE39bus()
    try
        args = (0, 0.01, 5000, 1500, 0.0)
        IEEE39bus_time_series(args)
        true
    catch
        false
    end
end

function testLoadIEEE39bus()
    try
        path = joinpath(@__DIR__, "../src/data/time_series/") 
        load(path * "IEEE39bus_time_series_seed_0000_diff_0.000_tanl_0.01" *
             "_nanl_05000_spin_1500_h_0.010.jld2")
        true
    catch
        false
    end
end


#######################################################################################################################
# end module 

end
