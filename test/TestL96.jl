########################################################################################################################
module TestL96
########################################################################################################################
########################################################################################################################
# imports and exports
using DataAssimilationBenchmarks.DeSolvers
using DataAssimilationBenchmarks.L96
using Test

########################################################################################################################
########################################################################################################################
# first testset using Euler Maruyama

function EMStep()
    # initial conditions and arguments
    x = zeros(40)

    # step size
    h = 0.01
    F = 8.0

    # forcing parameter
    dx_params = Dict{String, Array{Float64}}("F" => [F])

    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => 0.0,
        "dx_params" => dx_params,
        "dx_dt" => L96.dx_dt,
        )

    # parameters to test
    # em_step! writes over x in place
    em_step!(x, 0.0, kwargs)

    # evaluate test pass/fail if the vector of x is equal to (f*h) in every instance
    if sum(x .== (F*h)) == 40
        true
    else
        false
    end

end

########################################################################################################################

end
