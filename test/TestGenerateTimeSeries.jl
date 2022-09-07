##############################################################################################
module TestGenerateTimeSeries
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks
using DataAssimilationBenchmarks.SingleExperimentDriver
using DataAssimilationBenchmarks.GenerateTimeSeries
using JLD2, Random

##############################################################################################
# Test generation and loading of the L96 model time series in default localtion

function testGenL96()
    try
        L96_time_series(time_series_exps["L96_deterministic_test"])
        true
    catch
        false
    end
end

function testLoadL96()
    try
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"
        load(path * "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05" *
             "_nanl_05000_spin_1500_h_0.050.jld2")
        true
    catch
        false
    end
end


##############################################################################################
# Test generation and loading of the IEEE39 bus model time series in the default location

function testGenIEEE39bus()
    try
        IEEE39bus_time_series(time_series_exps["IEEE39bus_deterministic_test"])
        true
    catch
        false
    end
end

function testLoadIEEE39bus()
    try
        path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"
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
