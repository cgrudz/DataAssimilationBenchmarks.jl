########################################################################################################################
module runtests
########################################################################################################################
########################################################################################################################
# imports and exports
using DataAssimilationBenchmarks.DeSolvers
using DataAssimilationBenchmarks.L96
using JLD
using Random
using Test


include("TestDeSolvers.jl")
import .TestDeSolvers

#test case 1: TestDeSolvers
@testset "DeSolvers" begin
    @test TestDeSolvers.test1
end
end
