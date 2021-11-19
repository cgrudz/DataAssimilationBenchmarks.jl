########################################################################################################################
module TestDeSolvers
########################################################################################################################
########################################################################################################################
# imports and exports
using DataAssimilationBenchmarks.DeSolvers
using DataAssimilationBenchmarks.L96
using LsqFit
using Test
########################################################################################################################
########################################################################################################################
# Define test function for analytical verification of discretization error
VecA = Union{Vector{Float64}, SubArray{Float64, 1}}
ParamDict = Union{Dict{String, Array{Float64}}, Dict{String, Vector{Float64}}}

function exponentialODE(x::T, t::Float64, dx_params::ParamDict) where {T <: VecA}
    # fill in exponential function here in the form of the other time derivatives
    # like L96.dx_dt, where the function is in terms of dx/dt=e^(t) so that we can
    # compute the answer analytically for comparision

    return [exp(t)]
end


########################################################################################################################
# Define auxiliary function to compute the difference of the simulated integral versus the analytical value

function expDiscretizationError(step_model!, h)
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

    abs(x[1] - exp(tanl))

end


########################################################################################################################
# Define auxiliary function to compute the least-squares estimated order of convergence

function calculateOrderConvergence(step_model!)
    # set step sizes in increasing order for log-10 log-10 analysis
    h_range = [0.0001, 0.001, 0.01]
    error_range = Vector{Float64}(undef, 3)

    # loop the discretization and calculate the errors
    for i in 1:length(h_range)
        error_range[i] = expDiscretizationError(step_model!, h_range[i])
    end

    h_range_log10 = log10.(h_range)
    error_range_log10 = log10.(error_range)

    function model_lsq_squares(x,p)
        @.p[1] + p[2]*x
    end

    fit = curve_fit(model_lsq_squares, h_range_log10, error_range_log10, [1.0,1.0])
    return coef(fit)
    # Set the least squares estimate for the line in log-10 log-10 scale using LsqFit
    # where we have the h_range in the x-axis and the error_range in the
    # y-axis.  The order of convergence is defined by the slope of this line
    # whereas the intercept is a constant factor (not necessary to report here).
    # Extract the slope and return the slope here, rounded to the hundredth place

end



########################################################################################################################
# Wrapper function to be supplied to runtests

function testEMExponential()
    coef = calculateOrderConvergence(em_step!)

    if abs(coef[2] - 1.0) > 0.1
        return false
    else
        return true
    end
end


########################################################################################################################
# Wrapper function to be supplied to runtests

function testRKExponential()
    coef = calculateOrderConvergence(rk4_step!)

    if abs(coef[2] - 4.0) > 0.1
        return false
    else
        return true
    end
end


testRKExponential()

########################################################################################################################

end
