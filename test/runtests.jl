##############################################################################################
module runtests
##############################################################################################
# imports and exports
using Test
using JLD2
##############################################################################################
# include test sub-modules
include("TestObsOperators.jl")
include("Test3dVAR")
include("TestDeSolvers.jl")
include("TestL96.jl")
include("TestTimeSeriesGeneration.jl")
include("TestIEEE39bus.jl")
include("TestFilterExps.jl")
include("TestClassicSmootherExps.jl")
include("TestIterativeSmootherExps.jl")
include("TestSingleIterationSmootherExps.jl")
##############################################################################################
# Run tests

# test set 1: Calculate the order of convergence for standard integrators
@testset "Calculate Order Convergence" begin
    @test TestDeSolvers.testEMExponential()
    @test TestDeSolvers.testRKExponential()
end

# test set 2: test L96 model equations for known behavior
@testset "Lorenz-96" begin
    @test TestL96.Jacobian()
    @test TestL96.EMZerosStep()
    @test TestL96.EMFStep()
end

# test set 3: Test time series generation, saving output to default directory and loading
@testset "Time Series Generation" begin
    @test TestTimeSeriesGeneration.testGenL96()
    @test TestTimeSeriesGeneration.testLoadL96()
    @test TestTimeSeriesGeneration.testGenIEEE39bus()
    @test TestTimeSeriesGeneration.testLoadIEEE39bus()
end

# test set 4: test the model equations for known behavior
@testset "IEEE 39 Bus" begin
    @test TestIEEE39bus.test_synchrony()
end

# test set 5: test filter state and parameter experiments
@testset "Filter Experiments" begin
    @test TestFilterExps.run_filter_state_L96()
    @test TestFilterExps.analyze_filter_state_L96()
    @test TestFilterExps.run_filter_param_L96()
    @test TestFilterExps.analyze_filter_param_L96()
    @test TestFilterExps.run_filter_state_IEEE39bus()
    @test TestFilterExps.analyze_filter_state_IEEE39bus()
end

# test set 6: test classic smoother state and parameter experiments
@testset "Classic Smoother Experiments" begin
    @test TestClassicSmootherExps.run_smoother_state_L96()
    @test TestClassicSmootherExps.analyze_smoother_state_L96()
    @test TestClassicSmootherExps.run_smoother_param_L96()
    @test TestClassicSmootherExps.analyze_smoother_param_L96()
end

# test set 7: test IEnKS smoother state and parameter experiments
@testset "Iterative Smoother Experiments" begin
    @test TestIterativeSmootherExps.run_sda_smoother_state_L96()
    @test TestIterativeSmootherExps.analyze_sda_smoother_state_L96()
    @test TestIterativeSmootherExps.run_sda_smoother_param_L96()
    @test TestIterativeSmootherExps.analyze_sda_smoother_param_L96()
    @test TestIterativeSmootherExps.run_sda_smoother_state_L96()
    @test TestIterativeSmootherExps.analyze_sda_smoother_state_L96()
    @test TestIterativeSmootherExps.run_sda_smoother_param_L96()
    @test TestIterativeSmootherExps.analyze_sda_smoother_param_L96()
end

# test set 8: test SIEnKS smoother state and parameter experiments
@testset "Single Iteration Smoother Experiments" begin
    @test TestSingleIterationSmootherExps.run_sda_smoother_state_L96()
    @test TestSingleIterationSmootherExps.analyze_sda_smoother_state_L96()
    @test TestSingleIterationSmootherExps.run_sda_smoother_param_L96()
    @test TestSingleIterationSmootherExps.analyze_sda_smoother_param_L96()
    @test TestSingleIterationSmootherExps.run_mda_smoother_state_L96()
    @test TestSingleIterationSmootherExps.analyze_mda_smoother_state_L96()
    @test TestSingleIterationSmootherExps.run_mda_smoother_param_L96()
    @test TestSingleIterationSmootherExps.analyze_mda_smoother_param_L96()
end

# test set 9: test Observation Operators jacobian
@testset "Observation Operators" begin
    @test TestObsOperators.alternating_obs_jacobian_pos()
    @test TestObsOperators.alternating_obs_jacobian_zero()
    @test TestObsOperators.alternating_obs_jacobian_neg()
end

# test set 10: test 3D-VAR 
@testset "3DVAR" begin
    @test Test3dVAR.testCost()
    @test Test3dVAR.testGrad()
end

##############################################################################################
# end module

end
