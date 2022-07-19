##############################################################################################
module TestObsOperators
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks.ObsOperators
using ForwardDiff, ReverseDiff
##############################################################################################
"""
    alternating_obs_jacobian_pos() 

Tests the alternating observation operator jacobian function for known behavior with automatic 
differentiation using 'γ' > 1.0.
Returns whether the difference of computed jacobian is within error tolerance for every entry
"""
function alternating_obs_jacobian_pos()
    # 1-D ensemble argument
    x = [UnitRange{Float64}(1.0,40.0);]
    # observation dimension
    obs_dim = 20
    # test gammas
    gam_pos = Dict{String, Any}("γ" => 2.0)

    # wrapper function for γ > 1.0
    function wrap_pos(x)
        ObsOperators.alternating_obs_operator(x, obs_dim, gam_pos)
    end

    # jacobian computed via automatic differentiation 
    jacob_auto = ForwardDiff.jacobian(wrap_pos, x)

    # compute differences between ForwardDiff and ObsOperators calculated jacobians
    diff =  jacob_auto - ObsOperators.alternating_obs_operator_jacobian(x, obs_dim, gam_pos)

    # compare within error tolerance for every entry across all differences 
    if sum((abs.(diff)) .<= 0.001) == 20*40
        true
    else
        false
    end
end


##############################################################################################
"""
    alternating_obs_jacobian_zero() 

Tests the alternating observation operator jacobian function for known behavior with automatic 
differentiation using 'γ' == 0.0.
Returns whether the difference of computed jacobian is within error tolerance for every entry
"""
function alternating_obs_jacobian_zero()
    # 1-D ensemble argument
    x = [UnitRange{Float64}(1.0,40.0);]
    # observation dimension
    obs_dim = 20
    # test gamma (γ == 0.0)
    gam_zero = Dict{String, Any}("γ" => 0.0)

    # wrapper function for γ == 0.0
    function wrap_zero(x)
        ObsOperators.alternating_obs_operator(x, obs_dim, gam_zero)
    end

    # jacobian computed via automatic differentiation 
    jacob_auto = ForwardDiff.jacobian(wrap_zero, x)

    # compute difference between ForwardDiff and ObsOperators calculated jacobian
    diff = jacob_auto - ObsOperators.alternating_obs_operator_jacobian(x, obs_dim, gam_zero)

    # compare within error tolerance for every entry of difference matrix
    if sum((abs.(diff)) .<= 0.01) == 20*40
        true
    else
        false
    end
end


##############################################################################################
"""
    alternating_obs_jacobian_neg() 

Tests the alternating observation operator jacobian function for known behavior with automatic 
differentiation using 'γ' < 0.0.
Returns whether the difference of computed jacobian is within error tolerance for every entry
"""
function alternating_obs_jacobian_neg()
    # 1-D ensemble argument
    x = [UnitRange{Float64}(1.0,40.0);]
    # observation dimension
    obs_dim = 20
    # test gamma (γ < 0.0)
    gam_neg = Dict{String, Any}("γ" => -0.5)

    # wrapper function for γ < 0.0
    function wrap_neg(x)
        ObsOperators.alternating_obs_operator(x, obs_dim, gam_neg)
    end

    # jacobian computed via automatic differentiation 
    jacob_auto = ForwardDiff.jacobian(wrap_neg, x)

    # compute difference between ForwardDiff and ObsOperators calculated jacobian
    diff = jacob_auto - ObsOperators.alternating_obs_operator_jacobian(x, obs_dim, gam_neg)

    # compare within error tolerance for every entry of difference matrix
    if sum((abs.(diff)) .<= 0.001) == 20*40
        true
    else
        false
    end
end


##############################################################################################
# end module

end
