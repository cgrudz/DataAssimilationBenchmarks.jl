using DataAssimilationBenchmarks
using DataAssimilationBenchmarks.DeSolvers
using DataAssimilationBenchmarks.L96
import("./TestTimeSeriesGeneration.jl")

"""import some piece of data for reference (from artifacts) here"""

@testset "DataAssimilationBenchmarks.jl" begin
    my_data = my_time_series_experiment()
    logical_yes_no_array = my_data == my_refenence_data
    sum(logical_yes_no_array) == dimension_of_data
end
