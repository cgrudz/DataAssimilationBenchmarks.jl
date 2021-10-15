########################################################################################################################
module runtests
########################################################################################################################
########################################################################################################################
# imports and exports
using DataAssimilationBenchmarks.DeSolvers
using DataAssimilationBenchmarks.L96
using Test

########################################################################################################################
# include test sub-modules 
include("TestL96.jl")
include("TestTimeSeriesGeneration.jl")

########################################################################################################################
# Run tests

# test set 1: L96 model
@testset "Lorenz-96" begin
    @test TestL96.EMStep()
end

# test set 2: TestTimeSeriesGeneration
@testset "Time Series Save / Load" begin
    @test TestTimeSeriesGeneration.testL96()
end


########################################################################################################################

end
