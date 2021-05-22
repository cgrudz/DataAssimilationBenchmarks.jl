########################################################################################################################
module DeSolvers
########################################################################################################################
# imports and exports
using Debugger
using Random, Distributions
export rk4_step!, tay2_step!

########################################################################################################################
########################################################################################################################
# Type union declarations for multiple dispatch

# vectors and ensemble members of sample
VecA = Union{Vector{Float64}, SubArray{Float64, 1}}

########################################################################################################################
########################################################################################################################
# four-stage Runge-Kutta scheme

function rk4_step!(x::T, t::Float64, kwargs::Dict{String,Any}) where {T <: VecA}
    """One step of integration rule for l96 4 stage Runge-Kutta as discussed in Grudzien et al. 2020

    The rule has strong convergence order 1.0 for generic SDEs and order 4.0 for ODEs
    Arguments are given as
    x          -- array of a single state possibly including parameter values
    t          -- time point
    kwargs     -- this should include dx_dt, the paramters for the dx_dt and optional arguments
    dx_dt      -- time derivative function with arguments x and dx_params
    dx_params  -- tuple of parameters necessary to resolve dx_dt, not including parameters in the extended state vector 
    h          -- numerical discretization step size
    diffusion  -- tunes the standard deviation of the Wiener process, equal to sqrt(h) * diffusion
    state_dim  -- keyword for parameter estimation, dimension of the dynamic state < dimension of full extended state
    ξ          -- random array size state_dim, can be defined in kwargs to provide a particular realization
    """

    # unpack the integration scheme arguments and the parameters of the derivative
    h = kwargs["h"]::Float64
    diffusion = kwargs["diffusion"]::Float64
    dx_dt = kwargs["dx_dt"]

    if haskey(kwargs, "dx_params")
        # get parameters for resolving dx_dt
        params = kwargs["dx_params"]::Vector{Float64}
    end

    # infer the (possibly) extended state dimension
    @bp
    sys_dim = length(x)

    # check if extended state vector
    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]::Int64
        v = @view x[begin: state_dim]
        param_est = true
    else
        # the state dim equals the system dim
        state_dim = sys_dim
        v = @view x[begin: state_dim]
        param_est = false
    end

    # check if SDE formulation
    if diffusion != 0.0
	    if haskey(kwargs, "ξ")
            # pre-computed perturbation is provided for controlled run
	        ξ = kwargs["ξ"]::Array{Float64,2}
	    else
            # generate perturbation for brownian motion if not neccesary to reproduce
            ξ = rand(MvNormal(zeros(state_dim), 1.0)) 
        end

    else
        # if deterministic RK, load dummy ξ of zeros
        ξ = zeros(state_dim)
    end

    # rescale the standard normal to variance h for Wiener process
    W = ξ * sqrt(h)

    # load parameter values from the extended state into the derivative
    if param_est
        if haskey(kwargs, "dx_params")
            # extract the parameter sample and append to other derivative parameters
            params = [params[:]; x[state_dim + 1: end]]
        
        else
            # set the parameter sample as the only derivative parameters
            params = x[state_dim + 1: end]
        end
    end

    # terms of the RK scheme recursively evolve the dynamic state components alone
    k1 = dx_dt(v, params, t) * h + diffusion * W
    k2 = dx_dt(v + 0.5 * k1, params, t + 0.5 * h) * h + diffusion * W
    k3 = dx_dt(v + 0.5 * k2, params, t + 0.5 * h) * h + diffusion * W
    k4 = dx_dt(v + k3, params, t + h) * h + diffusion * W
    
    # compute the update to the dynamic variables
    x[begin: state_dim] = v + (1.0 / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4) 
    x
end

########################################################################################################################
# deterministic 2nd order Taylor Method

function tay2_step!(x::Vector{Float64}, t::Float64, kwargs::Dict{String,Any})
    """Second order Taylor method for step size h and state vector x.

    Arguments are given as
    x          -- array of a single state possibly including parameter values
    kwargs     -- this should include dx_dt, the paramters for the dx_dt and optional arguments
    dx_dt      -- time derivative function with arguments x and dx_params
    dx_params  -- tuple of parameters necessary to resolve dx_dt, not including parameters in the extended state vector 
    h          -- numerical discretization step size
    """

    # unpack dx_params
    h = kwargs["h"]::Float64
    params = kwargs["dx_params"]::Vector{Float64}
    dx_dt = kwargs["dx_dt"]
    jacobian = kwargs["jacobian"]

    # calculate the evolution of x one step forward via the second order Taylor expansion

    # first derivative
    dx = dx_dt(x, params, t)

    # second order taylor expansion
    x .= x + dx * h + 0.5 * jacobian(x, params, t) * dx * h^2.0
end

########################################################################################################################
# Euler-Murayama step

function em_step!(x::Vector{Float64}, t::Float64, kwargs::Dict{String,Any})
    """This will propagate the state x one step forward by Euler-Murayama

    Step size is h and the Wiener process is assumed to have a scalar diffusion coefficient"""

    # unpack the arguments for the integration step
    h = kwargs["h"]::Float64 
    params = kwargs["dx_params"]::Vector{Float64}
    diffusion = kwargs["diffusion"]::Float64
    dx_dt = kwargs["dx_dt"]
    state_dim = length(x)

    # check if SDE or deterministic formulation
    if diffusion != 0.0
	    if haskey(kwargs, "ξ")
            # pre-computed perturbation is provided for controlled run
	        ξ = kwargs["ξ"]::Array{Float64,2}
	    else
            # generate perturbation for brownian motion if not neccesary to reproduce
            ξ = rand(MvNormal(zeros(state_dim), 1.0)) 
        end

    else
        # if deterministic Euler, load dummy ξ of zeros
        ξ = zeros(state_dim)
    end

    # rescale the standard normal to variance h for Wiener process
    W = ξ * sqrt(h)

    # step forward by interval h
    x .= x +  h * dx_dt(x, params, t) + diffusion * W
end

########################################################################################################################
# end module

end
