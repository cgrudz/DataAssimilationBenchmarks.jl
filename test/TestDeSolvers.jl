##############################################################################################
module TestDeSolvers
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks
using DataAssimilationBenchmarks.DeSolvers, DataAssimilationBenchmarks.L96
using LsqFit
##############################################################################################
"""
    exponentialODE(x::T, t::Float64, dx_params::ParamDict) where {T <: VecA}

Wrapper for making a vectorized output of the exponential function for the DE solvers.  This
is used to verify the order of convergence for integration methods versus an analytical
solution.
"""
function exponentialODE(x::VecA(T), t::T, dx_params::ParamDict(T)) where T <: Float64
    [exp(t)]
end


##############################################################################################
"""
    expDiscretizationError(step_model!, h)

Auxiliary function to compute the difference of the numerically simulated integral versus
the analytical value.  This is a function of the time step and integration method, for
varying the approximations to demonstrate the correct reduction in discretization errors.
"""
function expDiscretizationError(step_model!, h::Float64)
    # continuous time length of the integration
    tanl = 0.1

    # discrete integration steps
    fore_steps = convert(Int64, tanl/h)
    time_steps = LinRange(0, tanl, fore_steps + 1)

    # initial data for the exponential function
    x = [1.0]

    # set the kwargs for the integration scheme
    # with empty values for the uneccessary parameters
    diffusion = 0.0
    dx_params = Dict{String, Array{Float64}}()
    kwargs = Dict{String, Any}(
            "h" => h,
            "diffusion" => diffusion,
            "dx_params" => dx_params,
            "dx_dt" => exponentialODE,
            )
    for i in 1:fore_steps
        step_model!(x, time_steps[i], kwargs)
    end

    # find the absolute difference of the apprximate integral from the built-in exponential
    abs(x[1] - exp(tanl))
end


##############################################################################################
"""
    calculateOrderConvergence(step_model!)

Auxiliary function to compute the least-squares estimated order of convergence for the
numerical integration schemes.  This ranges over step sizes as a function of the integration
method, and calculates the log-10 / log-10 slope and intercept for change in error with
respect to step size.
"""
function calculateOrderConvergence(step_model!)
    # set step sizes in increasing order for log-10 log-10 analysis
    h_range = [0.005, 0.01, 0.05, 0.1]
    error_range = Vector{Float64}(undef, length(h_range))

    # loop the discretization and calculate the errors
    for i in 1:length(h_range)
        error_range[i] = expDiscretizationError(step_model!, h_range[i])
    end

    # convert the error and the step sizes to log-10
    h_range_log10 = log10.(h_range)
    error_range_log10 = log10.(error_range)

    function model_lsq_squares(x,p)
        # define a function object to vary parameters p
        @. p[1] + p[2]*x
    end

    # fit the best-fit line and return coefficients
    fit = curve_fit(model_lsq_squares, h_range_log10, error_range_log10, [1.0, 1.0])
    coef(fit)
end


##############################################################################################
function testEMExponential()
    coef = calculateOrderConvergence(em_step!)

    if abs(coef[2] - 1.0) > 0.1
        false
    else
        true
    end
end


##############################################################################################
function testRKExponential()
    coef = calculateOrderConvergence(rk4_step!)

    if abs(coef[2] - 4.0) > 0.1
        false
    else
        true
    end
end


##############################################################################################
# end module

end
