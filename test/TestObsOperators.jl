##############################################################################################
module TestObsOperators
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks.ObsOperators
using ForwardDiff
##############################################################################################
"""
    alternating_obs_jacobian() 

Tests the alternating observation operator jacobian function for known behavior with automatic 
    differentiation.
Returns whether the difference of computed jacobians is within error tolerance for every entry
"""
function alternating_obs_jacobian()
    # 1-D ensemble argument
    x = [UnitRange{Float64}(1.0,40.0);]
    # observation dimension
    obs_dim = 20
    # test gammas
    gam_pos = Dict{String, Any}("γ" => 2.0)
    gam_zero = Dict{String, Any}("γ" => 0.0)
    gam_neg = Dict{String, Any}("γ" => -0.5)

    # wrapper functions for different gammas
    # γ > 1.0
    function wrap_pos(x)
        ObsOperators.alternating_obs_operator(x, obs_dim, gam_pos)
    end
    # γ == 0.0
    function wrap_zero(x)
        ObsOperators.alternating_obs_operator(x, obs_dim, gam_zero)
    end
    # γ < 0.0
    function wrap_neg(x)
        ObsOperators.alternating_obs_operator(x, obs_dim, gam_neg)
    end


    pos_jacob_auto = ObsOperators.alternating_projector(ForwardDiff.jacobian(wrap_pos, x), obs_dim)
    zero_jacob_auto = ObsOperators.alternating_projector(ForwardDiff.jacobian(wrap_zero, x), obs_dim)
    neg_jacob_auto = ObsOperators.alternating_projector(ForwardDiff.jacobian(wrap_neg, x), obs_dim)

    # compute differences between ForwardDiff and ObsOperators calculated jacobians
    diff_pos =  pos_jacob_auto - ObsOperators.alternating_obs_operator_jacobian(x, obs_dim, gam_pos)
    diff_zero = zero_jacob_auto - ObsOperators.alternating_obs_operator_jacobian(x, obs_dim, gam_zero)
    diff_neg = neg_jacob_auto - ObsOperators.alternating_obs_operator_jacobian(x, obs_dim, gam_neg)

    # compare within error tolerance for every entry across all differences 
    if sum((abs.(diff_pos)) .<= 0.01) != 20*40
        false
    elseif sum((abs.(diff_zero)) .<= 0.01) != 20*40
        false
    elseif sum((abs.(diff_neg)) .<= 0.01) != 20*40
        false
    else
        true
    end
end

##############################################################################################
# end module

end
