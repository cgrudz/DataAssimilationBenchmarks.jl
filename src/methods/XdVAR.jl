##############################################################################################
module XdVAR
##############################################################################################
# imports and exports
using LinearAlgebra, SparseArrays
using ..DataAssimilationBenchmarks
using ForwardDiff, ReverseDiff
##############################################################################################
# Main methods
##############################################################################################
"""
    D3_var_cost(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
"""

function D3_var_cost(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
    H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
    # initializations
    obs_dim = length(obs)
    # obs operator
    H = H_obs(x,obs_dim,kwargs)

    back_component = transpose((x - x_background))*(inv(state_cov))*(x - x_background)
    obs_component = transpose((obs - H))*(inv(obs_cov))*(obs - H)
    J = 0.5*back_component + 0.5*obs_component
    return J
end


##############################################################################################
"""
    D3_var_grad(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
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
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
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
    D3_var_hessian(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
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
        print("Iteration: " * string(j) * "\n")
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