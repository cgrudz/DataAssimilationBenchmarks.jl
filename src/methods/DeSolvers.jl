##############################################################################################
module DeSolvers
##############################################################################################
# imports and exports
using ..DataAssimilationBenchmarks
export rk4_step!, tay2_step!, em_step!
##############################################################################################
"""
    rk4_step!(x::VecA(T), t::Float64, kwargs::StepKwargs) where T <: Real

Steps model state with the 4 stage Runge-Kutta scheme.

The rule has strong convergence order 1.0 for generic SDEs and order 4.0 for ODEs.
This method overwrites the input in-place and returns the updated
```
return x
```
"""
function rk4_step!(x::VecA(T), t::Float64, kwargs::StepKwargs) where T <: Real
    # unpack the integration scheme arguments and the parameters of the derivative
    h = kwargs["h"]::Float64
    diffusion = kwargs["diffusion"]::Float64
    dx_dt = kwargs["dx_dt"]::Function

    if haskey(kwargs, "dx_params")
        # get parameters for resolving dx_dt
        dx_params = kwargs["dx_params"]::ParamDict(T)
    end

    # infer the (possibly) extended state dimension
    sys_dim = length(x)

    # check if extended state vector
    if haskey(kwargs, "param_sample")
        param_sample = kwargs["param_sample"]::ParamSample
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
            ξ = rand(Normal(), state_dim) 
        end
        if haskey(kwargs, "diff_mat")
            # diffusion is a scalar intensity which is applied to the 
            # structure matrix for the diffusion coefficients
            diff_mat = kwargs["diff_mat"]::Array{Float64}
            diffusion = diffusion * diff_mat 
        end
        # rescale the standard normal to variance h for Wiener process
        W = ξ * sqrt(h)
    end

    # load parameter values from the extended state into the derivative
    if param_est
        if haskey(kwargs, "dx_params")
            # extract the parameter sample and append to other derivative parameters
            for key in keys(param_sample)
                dx_params = merge(dx_params, Dict(key => x[param_sample[key][1]]))
            end
        else
            # set the parameter sample as the only derivative parameters
            dx_params = Dict{String, Array{T}}
            for key in keys(param_sample)
                dx_params = merge(dx_params, Dict(key => x[param_sample[key][1]]))
            end
        end
    end

    # pre-allocate storage for the Runge-Kutta scheme
    κ = Array{T}(undef, state_dim, 4)

    # terms of the RK scheme recursively evolve the dynamic state components alone
    if diffusion != 0.0
        # SDE formulation
        κ[:, 1] = dx_dt(v, t, dx_params) * h + diffusion * W
        κ[:, 2] = dx_dt(v + 0.5 * κ[:, 1], t + 0.5 * h, dx_params) * h + diffusion * W
        κ[:, 3] = dx_dt(v + 0.5 * κ[:, 2], t + 0.5 * h, dx_params) * h + diffusion * W
        κ[:, 4] = dx_dt(v + κ[:, 3], t + h, dx_params) * h + diffusion * W
    else
        # deterministic formulation
        κ[:, 1] = dx_dt(v, t, dx_params) * h 
        κ[:, 2] = dx_dt(v + 0.5 * κ[:, 1], t + 0.5 * h, dx_params) * h 
        κ[:, 3] = dx_dt(v + 0.5 * κ[:, 2], t + 0.5 * h, dx_params) * h 
        κ[:, 4] = dx_dt(v + κ[:, 3], t + h, dx_params) * h 
    end
    
    # compute the update to the dynamic variables
    x[begin: state_dim] = v + (1.0 / 6.0) * (κ[:, 1] + 2.0*κ[:, 2] + 2.0*κ[:, 3] + κ[:, 4]) 
    return x
end


##############################################################################################
"""
    tay2_step!(x::VecA(T), t::Float64, kwargs::StepKwargs) where T<: Real

Steps model state with the deterministic second order autonomous Taylor method.

This method has order 2.0 convergence for autonomous ODEs.
Time variable `t` is just a dummy variable, where this method is not defined for non-autonomous
dynamics. This overwrites the input in-place and returns the updated
```
return x
```
"""
function tay2_step!(x::VecA(T), t::Float64, kwargs::StepKwargs) where T <: Real
    # unpack dx_params
    h = kwargs["h"]::Float64
    dx_params = kwargs["dx_params"]::ParamDict(T)
    dx_dt = kwargs["dx_dt"]::Function
    jacobian = kwargs["jacobian"]::Function

    # calculate the evolution of x one step forward via the second order Taylor expansion

    # first derivative
    dx = dx_dt(x, t, dx_params)

    # second order taylor expansion
    x .= x + dx * h + 0.5 * jacobian(x, t, dx_params) * dx * h^2.0
    return x
end


##############################################################################################
"""
    em_step!(x::VecA(T), t::Float64, kwargs::StepKwargs) where T <: Real

Steps model state with the Euler-Maruyama scheme.

This method has order 1.0 convergence for ODEs and for SDEs with additive noise, though has
inferior performance to the four stage Runge-Kutta scheme when the amplitude of the SDE noise
purturbations are small-to-moderately large.

This overwrites the input in-place and returns the updated
```
return x
```
"""
function em_step!(x::VecA(T), t::Float64, kwargs::StepKwargs) where T <: Real
    # unpack the arguments for the integration step
    h = kwargs["h"]::Float64 
    dx_params = kwargs["dx_params"]::ParamDict(T)
    diffusion = kwargs["diffusion"]::Float64
    dx_dt = kwargs["dx_dt"]::Function
    state_dim = length(x)

    # check if SDE or deterministic formulation
    if diffusion != 0.0
	    if haskey(kwargs, "ξ")
            # pre-computed perturbation is provided for controlled run
	        ξ = kwargs["ξ"]::Array{Float64,2}
	    else
            # generate perturbation for brownian motion if not neccesary to reproduce
            ξ = rand(Normal(), state_dim) 
        end

    else
        # if deterministic Euler, load dummy ξ of zeros
        ξ = zeros(state_dim)
    end

    # rescale the standard normal to variance h for Wiener process
    W = ξ * sqrt(h)

    # step forward by interval h
    x .= x +  h * dx_dt(x, t, dx_params) + diffusion * W
    return x
end


##############################################################################################
# end module

end
