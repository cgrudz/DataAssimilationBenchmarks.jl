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

########################################################################################################################
########################################################################################################################

function exponentialODE(x::T, t::Float64, dx_params::ParamDict) where {T <: VecA}
    # Wrapper for making a vector form of the exponential function for the DE solvers
    [exp(t)]
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

    # find the absolute difference of the apprximate integral from the built-in exponential
    abs(x[1] - exp(tanl))
end


########################################################################################################################
# Define auxiliary function to compute the least-squares estimated order of convergence

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



########################################################################################################################
# Wrapper function to be supplied to runtests

function testEMExponential()
    coef = calculateOrderConvergence(em_step!)

    if abs(coef[2] - 1.0) > 0.1
        false
    else
        true
    end
end


########################################################################################################################
# Wrapper function to be supplied to runtests

function testRKExponential()
    coef = calculateOrderConvergence(rk4_step!)

    if abs(coef[2] - 4.0) > 0.1
        false
    else
        true
    end
end


########################################################################################################################

end
