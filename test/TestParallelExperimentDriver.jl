##############################################################################################
module TestParallelExperimentDriver
##############################################################################################
using DataAssimilationBenchmarks
using DataAssimilationBenchmarks.ParallelExperimentDriver

##############################################################################################
"""
    This is to test if the named tuple constructor will generate the experiment configuration

"""
function test_ensemble_filter_adaptive_inflation()
    try
        args, wrap_exp = ensemble_filter_adaptive_inflation()
        wrap_exp(args[1])
        true
    catch
        false
    end
end

##############################################################################################
"""
    This is to test if the named tuple constructor will generate the experiment configuration

"""
function test_D3_var_tuned_inflation()
    try
        args, wrap_exp = D3_var_tuned_inflation()
        wrap_exp(args[1])
        true
    catch
        false
    end
end

##############################################################################################
"""
    This is to test if the named tuple constructor will generate the experiment configuration

"""
function test_ensemble_filter_param()
    try
        args, wrap_exp = ensemble_filter_param()
        wrap_exp(args[1])
        true
    catch
        false
    end
end

##############################################################################################
"""
    This is to test if the named tuple constructor will generate the experiment configuration

"""
function test_classic_ensemble_state()
    try
        args, wrap_exp = classic_ensemble_state()
        wrap_exp(args[1])
        true
    catch
        false
    end
end

##############################################################################################
"""
    This is to test if the named tuple constructor will generate the experiment configuration

"""
function test_classic_ensemble_param()
    try
        args, wrap_exp = classic_ensemble_param()
        wrap_exp(args[1])
        true
    catch
        false
    end
end

##############################################################################################
"""
    This is to test if the named tuple constructor will generate the experiment configuration

"""
function test_single_iteration_ensemble_state()
    try
        args, wrap_exp = single_iteration_ensemble_state()
        wrap_exp(args[1])
        true
    catch
        false
    end
end

##############################################################################################
"""
    This is to test if the named tuple constructor will generate the experiment configuration

"""
function test_iterative_ensemble_state()
    try
        args, wrap_exp = iterative_ensemble_state()
        wrap_exp(args[1])
        true
    catch
        false
    end
end

##############################################################################################
# end module

end
