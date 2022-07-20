##############################################################################################
module TestL96
##############################################################################################
using DataAssimilationBenchmarks.DeSolvers, DataAssimilationBenchmarks.L96
using ForwardDiff
##############################################################################################
"""
    Jacobian()

Tests the L96 jacobian function for known behavior with automatic differentiation.
Returns whether the difference of computed jacobians is within error tolerance for every entry
"""
function Jacobian()
    # dummy time argument
    t = 0.0

    # forcing parameter
    F = 8.0
    dx_params = Dict{String, Array{Float64}}("F" => [F])

    # wrapper function
    function wrap_dx_dt(x)
        L96.dx_dt(x, t, dx_params)
    end

    # model state
    x = Vector{Float64}(1:40)

    # compute difference between ForwardDiff and L96 calculated jacobians
    diff = Matrix(ForwardDiff.jacobian(wrap_dx_dt, x) - Matrix(L96.jacobian(x, t, dx_params)))

    # compare within error tolerance for every entry
    if sum((abs.(diff)) .<= 0.01) == 40*40
        true
    else
        false
    end

end

##############################################################################################
"""
    EMZerosStep()

Tests the L96 derivative function for known behavior with Euler(-Maruyama) method.
The initial condition of zeros returns h * F in all components
"""
function EMZerosStep()
    # step size
    h = 0.01

    # forcing parameter
    F = 8.0
    dx_params = Dict{String, Array{Float64}}("F" => [F])

    # initial conditions and arguments
    x = zeros(40)

    # parameters to test
    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => 0.0,
        "dx_params" => dx_params,
        "dx_dt" => L96.dx_dt,
        )

    # em_step! writes over x in place
    em_step!(x, 0.0, kwargs)

    # evaluate test pass/fail if the vector of x is equal to (f*h) in every instance
    if sum(x .== (F*h)) == 40
        true
    else
        false
    end

end


##############################################################################################
"""
    EMFStep()

Tests the L96 derivative function for known behavior with Euler(-Maruyama) method.
The vector with all components equal to the forcing parameter F is a fixed point for the
system and the time derivative should be zero with this initiial condition.
"""
function EMFStep()
    # step size
    h = 0.01

    # forcing parameter
    F = 8.0
    dx_params = Dict{String, Array{Float64}}("F" => [F])

    # initial conditions and arguments
    x = ones(40)
    x = x * F

    # parameters to test
    kwargs = Dict{String, Any}(
        "h" => h,
        "diffusion" => 0.0,
        "dx_params" => dx_params,
        "dx_dt" => L96.dx_dt,
        )

    # em_step! writes over x in place
    em_step!(x, 0.0, kwargs)

    # evaluate test pass/fail if the vector of x is equal to (f*h) in every instance
    if sum(x .== (F)) == 40
        true
    else
        false
    end

end


##############################################################################################
# end module

end
