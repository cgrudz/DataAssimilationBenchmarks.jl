##############################################################################################
module Test3dVAR
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks.XdVAR, DataAssimilationBenchmarks.ObsOperators
using ForwardDiff, LinearAlgebra, Random, Distributions
##############################################################################################
"""
    testCost() 

    Tests the 3dVAR cost function for known behavior.
"""
function testCost()
    # initialization
    x = ones(40) * 0.5
    obs = zeros(40)
    x_background = ones(40)
    state_cov = I
    obs_cov = I
    params = Dict{String, Any}("γ" => 1.0)
    H_obs = alternating_obs_operator

    cost = XdVAR.D3_var_cost(x, obs, x_background, state_cov, H_obs, obs_cov, params)

    if abs(cost - 10) < 0.001
        true
    else
        false
    end
end

##############################################################################################
"""
    testGrad() 

    Tests the gradient 3dVAR cost function for known behavior using ForwardDiff.
"""
function testGrad()
    # initialization
    obs = zeros(40)
    x_background = ones(40)
    state_cov = I
    obs_cov = I
    params = Dict{String, Any}("γ" => 1.0)
    H_obs = alternating_obs_operator

    # wrapper function
    function wrap_cost(x)
        XdVAR.D3_var_cost(x, obs, x_background, state_cov, H_obs, obs_cov, params)
    end
    # input
    x = ones(40) * 0.5

    grad = ForwardDiff.gradient(wrap_cost, x)
    
    if norm(grad) < 0.001
        true
    else
        false
    end
end

##############################################################################################
"""
    testNewton() 

    Tests the Newton optimization of the 3dVAR cost function.
"""
function testNewton()
    # initialization
    x = 5.0 * ones(40)
    obs = zeros(40)
    x_background = ones(40)
    state_cov = I
    obs_cov = I
    params = Dict{String, Any}("γ" => 1.0)
    H_obs = alternating_obs_operator

    # perform Simple Newton optimization
    op = XdVAR.D3_var_NewtonOp(x, obs, x_background, state_cov, H_obs, obs_cov, params)
  
    if abs(sum(op - ones(40) * 0.5)) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    testNewtonNoise() 

    Tests the Newton optimization of the 3dVAR cost function with noise.
"""
function testNewtonNoise()
    # initialization
    x = ones(40)
    v = rand(Normal(0, 1), 40)
    obs = zeros(40) + v
    x_background = zeros(40)
    state_cov = I
    obs_cov = I
    params = Dict{String, Any}("γ" => 1.0)
    H_obs = alternating_obs_operator

    # perform Simple Newton optimization
    op = XdVAR.D3_var_NewtonOp(x, obs, x_background, state_cov, H_obs, obs_cov, params)
    
    if abs(sum(op - obs * 0.5)) < 0.001
        true
    else
        false
    end
end


##############################################################################################
# end module

end
