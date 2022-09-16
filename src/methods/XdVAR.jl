##############################################################################################
module XdVAR
##############################################################################################
# imports and exports
using LinearAlgebra, SparseArrays
using ..DataAssimilationBenchmarks
using ForwardDiff
##############################################################################################
# Main methods
##############################################################################################
"""
<<<<<<< Updated upstream
    D3_var_cost(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
=======
<<<<<<< Updated upstream
    D3_var_cost(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
>>>>>>> Stashed changes
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
=======
    D3_var_cost(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real

Computes the cost of the three-dimensional variational analysis increment from an initial state 
proposal with a static background covariance

'x' is the initial state proposal vector, 'obs' is to the observation vector, 'x_bkg'
is a free argument used to evaluate the cost of the given state proposal versus other proposal 
states, state_cov is the background error covariance matrix, H_obs is a model mapping operator 
for observations, and obs_cov is the observation error covariance matrix. 'kwargs' refers to 
any additional arguments needed for the operation computation.

Note: The observational and background components are equally weighted. 

```
return  0.5*back_component + 0.5*obs_component
```
>>>>>>> Stashed changes
"""
function D3_var_cost(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
    H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
    # initializations
    obs_dim = length(obs)
    # obs operator
<<<<<<< Updated upstream
    H = H_obs(x,obs_dim,kwargs)
=======
    H = H_obs(x, obs_dim, kwargs)

    # background discrepancy
    δ_b = x - x_bkg

    # observation discrepancy
    δ_o = obs - H
>>>>>>> Stashed changes

    back_component = transpose((x - x_background))*(inv(state_cov))*(x - x_background)
    obs_component = transpose((obs - H))*(inv(obs_cov))*(obs - H)
    J = 0.5*back_component + 0.5*obs_component
    return J
end


##############################################################################################
"""
<<<<<<< Updated upstream
    D3_var_grad(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
=======
<<<<<<< Updated upstream
    D3_var_grad(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
>>>>>>> Stashed changes
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
=======
    D3_var_grad(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T), 
                obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64

Computes the gradient of the three-dimensional variational analysis increment from an initial 
state proposal with a static background covariance using a wrapper function for automatic 
differentiation

'x' is the initial state proposal vector, 'obs' is to the observation vector, 'x_bkg'
is a free argument used to evaluate the cost of the given state proposal versus other proposal 
states, state_cov is the background error covariance matrix, H_obs is a model mapping operator 
for observations, and obs_cov is the observation error covariance matrix. 'kwargs' refers to 
any additional arguments needed for the operation computation.

```
return  ForwardDiff.gradient(wrap_cost, x)
```
>>>>>>> Stashed changes
"""

function D3_var_grad(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
    H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
    # initializations
    function wrap_cost(x)
        XdVAR.D3_var_cost(x, obs, x_background, state_cov, H_obs, obs_cov, kwargs)
    end

    grad = ForwardDiff.gradient(wrap_cost, x)
    return grad
end


##############################################################################################
"""
<<<<<<< Updated upstream
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
=======
<<<<<<< Updated upstream
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
>>>>>>> Stashed changes
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
=======
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                   obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64 

Computes the hessian of the three-dimensional variational analysis increment from an initial 
state proposal with a static background covariance using a wrapper function for automatic 
differentiation

'x' is the initial state proposal vector, 'obs' is to the observation vector, 'x_bkg'
is a free argument used to evaluate the cost of the given state proposal versus other proposal 
states, state_cov is the background error covariance matrix, H_obs is a model mapping operator 
for observations, and obs_cov is the observation error covariance matrix. 'kwargs' refers to 
any additional arguments needed for the operation computation.

```
return  ForwardDiff.hessian(wrap_cost, x)
```
>>>>>>> Stashed changes
"""

function D3_var_hessian(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
    H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real

    function wrap_cost(x)
        XdVAR.D3_var_cost(x, obs, x_background, state_cov, H_obs, obs_cov, kwargs)
    end

    hess = ForwardDiff.hessian(wrap_cost, x)

    return hess
end

##############################################################################################
"""
<<<<<<< Updated upstream
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
=======
<<<<<<< Updated upstream
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
>>>>>>> Stashed changes
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
=======
    D3_var_NewtonOp(x_bkg::VecA(T), obs::VecA(T), state_cov::CovM(T), obs_cov::CovM(T),
                    kwargs::StepKwargs) where T <: Float64

Computes the local minima of the three-dimension variational cost function with a static 
background covariance using a simple Newton optimization method

'x_bkg' is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, 'obs' is to the observation vector, state_cov is the background error 
covariance matrix, H_obs is a model mapping operator for observations, and obs_cov is the 
observation error covariance matrix. 'kwargs' refers to any additional arguments needed for 
the operation computation, and 'x' is the initial state proposal vector.

```
return  x
```
>>>>>>> Stashed changes
"""

function D3_var_NewtonOp(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T),
    H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
    # initializations
    j_max = 40
    tol = 0.001
    j = 1
    sys_dim = length(x)

    # gradient preallocation over-write
    function grad!(g::VecA(T), x::VecA(T)) where T <: Real
        g[:] = D3_var_grad(x, obs, x_background, state_cov, H_obs, obs_cov, kwargs)
    end

    # hessian preallocation over-write
    function hess!(h::ArView(T), x::VecA(T)) where T <: Real
        h .= D3_var_hessian(x, obs, x_background, state_cov, H_obs, obs_cov, kwargs)
    end

    # step 6: perform the optimization by simple Newton
    grad_x = Array{Float64}(undef, (sys_dim))
    hess_x = Array{Float64}(undef, (sys_dim, sys_dim))

    while j <= j_max
        #print("Iteration: " * string(j) * "\n")
        # compute the gradient and hessian
        grad!(grad_x, x)
        hess!(hess_x, x)

        # perform Newton approximation
        Δx = inv(hess_x)*grad_x
        x = x - Δx

        if norm(Δx) < tol
            break
        else
            j+=1
        end
    end
    return x
end

##############################################################################################
# end module

end