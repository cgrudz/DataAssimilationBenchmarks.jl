##############################################################################################
module DeSolvers
##############################################################################################
# imports and exports
using ..DataAssimilationBenchmarks
export rk4_step!, tay2_step!, em_step!
##############################################################################################
"""
    rk4_step!(x::VecA, t::Float64, kwargs::StepKwargs) 

Step of integration rule for 4 stage Runge-Kutta as discussed in Grudzien et al. 2020.
The rule has strong convergence order 1.0 for generic SDEs and order 4.0 for ODEs.
Arguments are given as:
```
    x      -- type [`VecA`](@ref) of model states possibly including static
              parameter values
    t      -- time value for present model state
    kwargs -- includes state time derivative dx_dt, paramters for the dx_dt
              and optionals
```
where `kwargs` is type [`StepKwargs`](@ref). Details on this scheme are available in the
manuscript
[Grudzien, C. et al. (2020).](https://gmd.copernicus.org/articles/13/1903/2020/gmd-13-1903-2020.html)
"""
function rk4_step!(x::VecA, t::Float64, kwargs::StepKwargs)
    # unpack the integration scheme arguments and the parameters of the derivative
    h = kwargs["h"]::Float64
    diffusion = kwargs["diffusion"]::Float64
    dx_dt = kwargs["dx_dt"]

    if haskey(kwargs, "dx_params")
        # get parameters for resolving dx_dt
        dx_params = kwargs["dx_params"]::ParamDict
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
            dx_params = Dict{String, Array{Float64}}
            for key in keys(param_sample)
                dx_params = merge(dx_params, Dict(key => x[param_sample[key][1]]))
            end
        end
    end

    # pre-allocate storage for the Runge-Kutta scheme
    κ = Array{T where T <: Real}(undef, state_dim, 4)

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
    x
end


##############################################################################################
"""
    tay2_step!(x::VecA, t::Float64, kwargs::StepKwargs) 

Deterministic second order autonomous Taylor method for step size `h` and state vector `x`.
Time variable `t` is just a dummy variable, where this method is not defined for non-autonomous
dynamics.  Arguments are given as:
```
    x      -- type [`VecA`](@ref) of model states possibly including static
              parameter values
    kwargs -- includes state time derivative dx_dt, paramters for the dx_dt
              and optionals
```
where `kwargs` is type [`StepKwargs`](@ref).
"""
function tay2_step!(x::VecA, t::Float64, kwargs::StepKwargs)
    # unpack dx_params
    h = kwargs["h"]::Float64
    dx_params = kwargs["dx_params"]::ParamDict
    dx_dt = kwargs["dx_dt"]
    jacobian = kwargs["jacobian"]

    # calculate the evolution of x one step forward via the second order Taylor expansion

    # first derivative
    dx = dx_dt(x, t, dx_params)

    # second order taylor expansion
    x .= x + dx * h + 0.5 * jacobian(x, t, dx_params) * dx * h^2.0
end


##############################################################################################
"""
    em_step!(x::VecA, t::Float64, kwargs::StepKwargs) 

This will propagate the state `x` one step forward by Euler-Maruyama scheme.
Arguments are given as:
```
    x      -- type [`VecA`](@ref) of model states possibly including static
              parameter values
    t      -- time value for present model state
    kwargs -- includes state time derivative dx_dt, paramters for the dx_dt
              and optionals
```
where `kwargs` is type [`StepKwargs`](@ref) Details on this scheme are available in the
manuscript
[Grudzien, C. et al.: (2020).](https://gmd.copernicus.org/articles/13/1903/2020/gmd-13-1903-2020.html)
"""
function em_step!(x::VecA, t::Float64, kwargs::StepKwargs)
    # unpack the arguments for the integration step
    h = kwargs["h"]::Float64 
    dx_params = kwargs["dx_params"]::ParamDict
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
end


##############################################################################################
# end module

end
