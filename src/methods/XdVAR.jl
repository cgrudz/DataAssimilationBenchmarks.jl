##############################################################################################
module XdVAR
##############################################################################################
# imports and exports
using LinearAlgebra, ForwardDiff
using ..DataAssimilationBenchmarks
export D3_var_cost, D3_var_grad, D3_var_hessian, D3_var_NewtonOp  
##############################################################################################
# Main methods
##############################################################################################
"""
    D3_var_cost(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real

Computes the cost of the three-dimensional variational analysis increment from an initial state 
proposal with a static background covariance

'x' is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, 'obs' is to the observation vector, 'x_bkg' is the initial state proposal 
vector, state_cov is the background error covariance matrix, H_obs is a model mapping operator 
for observations, and obs_cov is the observation error covariance matrix. 'kwargs' refers to 
any additional arguments needed for the operation computation.

```
return  0.5*back_component + 0.5*obs_component
```
"""
function D3_var_cost(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                     H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real

    # initializations
    obs_dim = length(obs)

    # obs operator
    H = H_obs(x, obs_dim, kwargs)

    # background discepancy
    δ_b = x - x_bkg

    # observation discrepancy
    δ_o = obs - H

    # cost function
    back_component = dot(δ_b, inv(state_cov) * δ_b)
    obs_component = dot(δ_o, inv(obs_cov) * δ_o)

    0.5*back_component + 0.5*obs_component
end


##############################################################################################
"""
    D3_var_grad(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64

Computes the gradient of the three-dimensional variational analysis increment from an initial 
state proposal with a static background covariance using a wrapper function for automatic 
differentiation

'x' is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, 'obs' is to the observation vector, 'x_bkg' is the initial state proposal 
vector, state_cov is the background error covariance matrix, H_obs is a model mapping operator 
for observations, and obs_cov is the observation error covariance matrix. 'kwargs' refers to 
any additional arguments needed for the operation computation.

'wrap_cost' is a function that allows differentiation with respect to the free argument 'x'
while treating all other hyperparameters of the cost function as constant.

```
return  ForwardDiff.gradient(wrap_cost, x)
```
"""
function D3_var_grad(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                     H_obs::Function, obs_cov::CovM(T),
                     kwargs::StepKwargs) where T <: Float64

    function wrap_cost(x::VecA(T)) where T <: Real
        D3_var_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    ForwardDiff.gradient(wrap_cost, x)
end


##############################################################################################
"""
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
        H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64 

Computes the hessian of the three-dimensional variational analysis increment from an initial 
state proposal with a static background covariance using a wrapper function for automatic 
differentiation

'x' is a free argument used to evaluate the cost of the given state proposal versus other 
proposal states, 'obs' is to the observation vector, 'x_bkg' is the initial state proposal 
vector, state_cov is the background error covariance matrix, H_obs is a model mapping 
operator for observations, and obs_cov is the observation error covariance matrix. 'kwargs' 
refers to any additional arguments needed for the operation computation.

'wrap_cost' is a function that allows differentiation with respect to the free argument 'x'
while treating all other hyperparameters of the cost function as constant.

```
return  ForwardDiff.hessian(wrap_cost, x)
```
"""
function D3_var_hessian(x::VecA(T), obs::VecA(T), x_bkg::VecA(T), state_cov::CovM(T),
                        H_obs::Function, obs_cov::CovM(T),
                        kwargs::StepKwargs) where T <: Float64 

    function wrap_cost(x::VecA(T)) where T <: Real
        D3_var_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    ForwardDiff.hessian(wrap_cost, x)
end


##############################################################################################
"""
    D3_var_NewtonOp(x_bkg::VecA(T), obs::VecA(T), state_cov::CovM(T), H_obs::Function,
        obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64

Computes the local minima of the three-dimension variational cost function with a static 
background covariance using a simple Newton optimization method

'x_bkg' is the initial state proposal vector, 'obs' is to the observation vector, state_cov is
the background error covariance matrix, H_obs is a model mapping operator for observations, 
obs_cov is the observation error covariance matrix, and 'kwargs' refers to any additional 
arguments needed for the operation computation.

```
return  x
```
"""
function D3_var_NewtonOp(x_bkg::VecA(T), obs::VecA(T), state_cov::CovM(T), H_obs::Function,
                         obs_cov::CovM(T), kwargs::StepKwargs) where T <: Float64 

    # initializations
    j_max = 40
    tol = 0.001
    j = 1
    sys_dim = length(x_bkg)

    # first guess is copy of the first background
    x = copy(x_bkg)

    # gradient preallocation over-write
    function grad!(g::VecA(T), x::VecA(T)) where T <: Real
        g[:] = D3_var_grad(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    # hessian preallocation over-write
    function hess!(h::ArView(T), x::VecA(T)) where T <: Real
        h .= D3_var_hessian(x, obs, x_bkg, state_cov, H_obs, obs_cov, kwargs)
    end

    # perform the optimization by simple Newton
    grad_x = Array{Float64}(undef, sys_dim)
    hess_x = Array{Float64}(undef, sys_dim, sys_dim)

    while j <= j_max
        #print("Iteration: " * string(j) * "\n")
        # compute the gradient and hessian
        grad!(grad_x, x)
        hess!(hess_x, x)

        # perform Newton approximation
        Δx = inv(hess_x) * grad_x
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