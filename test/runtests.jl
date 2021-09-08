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

########################################################################################################################
########################################################################################################################
# first testset using Euler Maruyama
@testset "Euler Maruyama" begin
    # initial conditions and arguments
    x = zeros(40)

    # step size
    h = 0.01

    # forcing parameter
    f = 8.0
    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => 0.0,
        "dx_params" => [f],
        "dx_dt" => L96.dx_dt,
        )

    # parameters to test
    # em_step! writes over x in place
    em_step!(x, 0.0, kwargs)

    # evaluate test pass/fail if the vector of x is equal to (f*h) in every instance
    @test sum(x .== (f*h)) == 40
end

########################################################################################################################
@testset "Time Series Output Test" begin
    try
        save(path * fname, data)
        did_write = true
    catch
        did_write = false
    end
    @test did_write
end
end
