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

function exponentialODE(x, t, dx_params)
    # fill in exponential function here in the form of the other time derivatives
    # like L96.dx_dt, where the function is in terms of dx/dt=e^(t) so that we can
    # compute the answer analytically for comparision
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
    x = 1.0

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

    abs(x - exp(tanl))
end


########################################################################################################################
# Define auxiliary function to compute the least-squares estimated order of convergence

function calculateOrderConvergence(step_model!)
    # set step sizes in increasing order for log-10 log-10 analysis
    h_range = .1 .^[3, 2, 1]
    error_range = Vector{Float64}(undef, 3)
    
    # loop the discretization and calculate the errors
    for i in 1:length(h_range)
        error_range[i] = expDiscretizationError(step_model!, h_range[i])
    end

    # Set the least squares estimate for the line in log-10 log-10 scale using LsqFit 
    # where we have the h_range in the x-axis and the error_range in the
    # y-axis.  The order of convergence is defined by the slope of this line
    # whereas the intercept is a constant factor (not necessary to report here).
    # Extract the slope and return the slope here, rounded to the hundredth place

end


########################################################################################################################
# Wrapper function to be supplied to runtests

function testEMExponential()
    slope = calculateOrderConvergence(em_step!)
    
    if abs(slope - 1.0) > 0.1
        false
    else
        true
    end
end


########################################################################################################################
# Wrapper function to be supplied to runtests

function testRKExponential()
    slope = calculateOrderConvergence(em_step!)
    
    if abs(slope - 4.0) > 0.1
        false
    else
        true
    end
end


########################################################################################################################

end
