#######################################################################################################################
module EnsembleKalmanSchemes
########################################################################################################################
########################################################################################################################
# imports and exports
using Debugger
using Random, Distributions, Statistics
using LinearAlgebra, Optim, SparseArrays
export alternating_obs_operator, analyze_ensemble, analyze_ensemble_parameters, rand_orth, inflate_state!,
       inflate_param!, transform, square_root, ensemble_filter, ls_smoother_classic, ls_smoother_hybrid, ls_smoother_hybrid_adaptive,
       ls_smoother_iterative

########################################################################################################################
########################################################################################################################
# Type union declarations for multiple dispatch

# covariance matrix types
CovM = Union{UniformScaling{Float64}, Diagonal{Float64}, Symmetric{Float64}}

# conditioning matrix types
ConM = Union{UniformScaling{Float64}, Symmetric{Float64}}

# observation matrix types
ObsH = Union{UniformScaling{Float64}, Diagonal{Float64}, Array{Float64}}

# transform matrix types
TransM = Union{Tuple{Symmetric{Float64,Array{Float64,2}},Array{Float64,2},Array{Float64,2}}, Array{Float64,2}}

########################################################################################################################
########################################################################################################################
# Main methods, debugged and validated
########################################################################################################################
# alternating id obs

function alternating_obs_operator(sys_dim::Int64, obs_dim::Int64, kwargs::Dict{String,Any})
    """Defines observation operator by alternating state vector components.

    If obs_dim == state_dim, this returns the identity matrix, otherwise alternating observations of the state
    components.  For parameter estimation, state_dim is an optional kwarg to define the operator to only observe
    the regular state vector, not the extended one."""

    if haskey(kwargs, "state_dim")
        # performing parameter estimation, load the dynamic state dimension
        state_dim = kwargs["state_dim"]::Int64
        
        # load observation operator for the extended state, without observing extended state components
        H = Matrix(1.0I, state_dim, sys_dim)
        
        # proceed with alternating observations of the regular state vector
        sys_dim = state_dim

    else
        if sys_dim == obs_dim
            H = 1.0I
        else
            H = Matrix(1.0I, sys_dim, sys_dim)
        end
    end

    if sys_dim == obs_dim
        return H

    elseif (obs_dim / sys_dim) > 0.5
        # the observation dimension is greater than half the state dimension, so we
        # remove only the trailing odd-index rows from the identity matrix, equal to the difference
        # of the state and observation dimension
        R = sys_dim - obs_dim
        H = vcat(H[1:end-2*R,:], H[end-2*R+2:2:end,:])

    elseif (obs_dim / sys_dim) == 0.5
        # the observation dimension is equal to half the state dimension so we remove exactly
        # half the rows, corresponding to those with even-index
        H = H[1:2:end,:]

    else
        # the observation dimension is less than half of the state dimension so that we
        # remove all even rows and then all but the remaining, leading obs_dim rows
        H = H[1:2:end,:]
        H = H[1:obs_dim,:]
    end
end


########################################################################################################################
# ensemble state statistics

function analyze_ensemble(ens::Array{Float64,2}, truth::Vector{Float64})
    """This will compute the ensemble RMSE as compared with the true twin, and the ensemble spread."""

    # infer the shapes
    sys_dim, N_ens = size(ens)

    # compute the ensemble mean
    x_bar = mean(ens, dims=2)

    # compute the RMSE of the ensemble mean
    rmse = sqrt(mean( (truth - x_bar).^2.0))

    # we compute the spread as in whitaker & louge 98 by the standard deviation 
    # of the mean square deviation of the ensemble from its mean
    spread = sqrt( ( 1.0 / (N_ens - 1.0) ) * sum(mean((ens .- x_bar).^2.0, dims=1)))

    return [rmse, spread]
end


########################################################################################################################
# ensemble parameter statistics

function analyze_ensemble_parameters(ens::Array{Float64,2}, truth::Vector{Float64})
    """This will compute the ensemble RMSE as compared with the true twin, and the ensemble spread."""

    # infer the shapes
    param_dim, N_ens = size(ens)

    # compute the ensemble mean
    x_bar = mean(ens, dims=2)

    # compute the RMSE of the ensemble mean, where each value is computed relative to the magnitude of the parameter
    rmse = sqrt( mean( (truth - x_bar).^2.0 ./ truth.^2.0 ) )

    # we compute the spread as in whitaker & louge 98 by the standard deviation of the mean square deviation of the 
    # ensemble from its mean, with the weight by the size of the parameter square
    spread = sqrt( ( 1.0 / (N_ens - 1.0) ) * sum(mean( (ens .- x_bar).^2.0 ./ 
                                                            (ones(param_dim, N_ens) .* truth.^2.0), dims=1)))
    
    return [rmse, spread]
end


########################################################################################################################
# random mean preserving orthogonal matrix, auxilliary function for determinstic EnKF schemes

function rand_orth(N_ens::Int64)
    """This generates a mean preserving random orthogonal matrix as in sakov oke 08"""
    
    Q = rand(Normal(), N_ens - 1, N_ens - 1)
    Q, R = qr!(Q)
    U_p =  zeros(N_ens, N_ens)
    U_p[1, 1] = 1.0
    U_p[2:end, 2:end] = Q

    b_1 = ones(N_ens) / sqrt(N_ens)
    Q = rand(Normal(), N_ens - 1, N_ens - 1)
    B = zeros(N_ens, N_ens)
    B[:, 1] = b_1
    B, R = qr!(B)
    B * U_p * transpose(B)
end


########################################################################################################################
# dynamic state variable inflation

function inflate_state!(ens::Array{Float64,2}, inflation::Float64, sys_dim::Int64, state_dim::Int64)
    """State variables are assumed to be in the leading rows, while extended
    state variables, parameter variables are after.
    
    Multiplicative inflation is performed only in the leading components."""

    if inflation == 1.0
        return ens
    else
        X_mean = mean(ens, dims=2)
        A = ens .- X_mean
        infl =  Matrix(1.0I, sys_dim, sys_dim) 
        infl[1:state_dim, 1:state_dim] .*= inflation 
        X_mean .+ infl * A
    end
end


########################################################################################################################
# parameter multiplicative inflation

function inflate_param!(ens::Array{Float64,2}, inflation::Float64, sys_dim::Int64, state_dim::Int64)
    """State variables are assumed to be in the leading rows, while extended
    state, parameter variables are after.
    
    Multiplicative inflation is performed only in the trailing components."""

    if inflation == 1.0
        return ens
    else
        X_mean = mean(ens, dims=2)
        A = ens .- X_mean
        infl =  Matrix(1.0I, sys_dim, sys_dim) 
        infl[state_dim+1: end, state_dim+1: end] .*= inflation
        X_mean .+ infl * A
    end
end


########################################################################################################################
# auxiliary function for square roots of multiple types of covariance matrices wrapped 

function square_root(M::T) where {T <: CovM}
    
    if T <: UniformScaling
        M^0.5
    elseif T <: Diagonal
        sqrt(M)
    else
        # stable square root for close-to-singular inverse calculations
        F = svd(M)
        Symmetric(F.U * Diagonal(sqrt.(F.S)) * F.Vt)
    end
end


########################################################################################################################
# transform auxilliary function for EnKF, ETKF, EnKS, ETKS

function transform(analysis::String, ens::Array{Float64,2}, H::T1, obs::Vector{Float64}, 
                   obs_cov::T2; conditioning::T3=0.0001I, 
                   m_err::Array{Float64,2}=(1.0 ./ zeros(1,1))) where {T1 <: ObsH, T2 <: CovM, T3 <: ConM}
    """Computes transform and related values for various flavors of ensemble Kalman schemes below.

    "analysis" is a string which determines the type of transform update.  The observation error covariance should be
    of UniformScaling, Diagonal or Symmetric type."""

    if analysis=="enkf" || analysis=="enks"
        ## This computes the stochastic transform for the EnKF/S as in Carrassi, et al. 2018
        # step 0: infer the ensemble, obs, and state dimensions
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the ensemble mean
        X_mean = mean(ens, dims=2)

        # step 2: compute the normalized anomalies
        A = (ens .- X_mean) / sqrt(N_ens - 1.0)

        # step 3: generate the unbiased perturbed observations, note, we use the actual observation error
        # covariance instead of the ensemble-based covariance to handle rank degeneracy
        obs_perts = rand(MvNormal(zeros(obs_dim), obs_cov), N_ens)
        obs_perts = obs_perts .- mean(obs_perts, dims=2)

        # step 4: compute the observation ensemble
        obs_ens = obs .+ obs_perts

        # step 5: generate the ensemble transform matrix, note, transform is missing normalization
        # of sqrt(N_ens-1) in paper
        Y = H * A
        C = Symmetric(Y * transpose(Y) + obs_cov)
        transform = 1.0I + transpose(Y) * inv(C) * (obs_ens - H * ens) / sqrt(N_ens - 1.0)
        
    elseif analysis=="etkf" || analysis=="etks"
        ## This computes the transform of the ETKF update as in Asch, Bocquet, Nodet
        ## This is the default method for the ensemble square root transform, given best
        ## performance metrics by BenchmarkTools
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the ensemble mean
        x_mean = mean(ens, dims=2)

        # step 2: compute the normalized anomalies
        A = (ens .- x_mean) / sqrt(N_ens - 1.0)
        
        # step 3: compute the ensemble in observation space
        Z = H * ens

        # step 4: compute the ensemble mean in observation space
        y_mean = mean(Z, dims=2)
        
        # step 5: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = inv(square_root(obs_cov))
        
        # then compute the weighted anomalies
        S = (Z .- y_mean) / sqrt(N_ens - 1.0)
        S = obs_sqrt_inv * S

        # step 6: compute the weighted innovation
        δ = obs_sqrt_inv * ( obs - y_mean )
       
        # step 7: compute the transform matrix
        T = inv(Symmetric(1.0I + transpose(S) * S))
        
        # step 8: compute the analysis weights
        w = T * transpose(S) * δ

        # step 9: compute the square root of the transform
        T_sqrt = sqrt(T)
        
        # step 10:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

        # step 11: package the transform output tuple
        T_sqrt, w, U
    
    elseif analysis=="etkf-svd" || analysis=="etks-svd"
        ## This computes the transform of the ETKF update as in Asch, Bocquet, Nodet
        ## This is an SVD-based variant on the direct ETKF above, BenchmarkTools gives
        ## this min, mean and median longer timing, though shorter max time
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the ensemble mean
        x_mean = mean(ens, dims=2)

        # step 2: compute the normalized anomalies
        A = (ens .- x_mean) / sqrt(N_ens - 1.0)
        
        # step 3: compute the ensemble in observation space
        Z = H * ens

        # step 4: compute the ensemble mean in observation space
        y_mean = mean(Z, dims=2)
        
        # step 5: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = inv(square_root(obs_cov))
        
        # then compute the weighted anomalies
        S = (Z .- y_mean) / sqrt(N_ens - 1.0)
        S = obs_sqrt_inv * S

        # step 6: compute the weighted innovation
        δ = obs_sqrt_inv * ( obs - y_mean )
       
        # step 7: compute the transform matrix
        F = svd(S)
        T = F.V * Diagonal( 1.0 ./ (1.0 .+ F.S.^2.0)) 
        
        # step 8: compute the analysis weights
        w = T * Diagonal(F.S) * transpose(F.U) * δ

        # step 9: compute the square root of the transform
        T_sqrt = Symmetric(F.V * Diagonal( sqrt.(1.0 ./ (1.0 .+ F.S.^2.0)) ) * F.Vt)
        
        # step 10:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

        # step 11: package the transform output tuple
        T_sqrt, w, U
    
    elseif analysis=="etkf_sqrt_core" || analysis=="etks_sqrt_core"
        ## This computes the transform of the ETKF update as in Asch, Bocquet, Nodet
        # but using a computation of the contribution of the model error covariance matrix Q
        # in the square root as in Raanes et al. 2015
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the ensemble mean
        x_mean = mean(ens, dims=2)

        # step 2a: compute the normalized anomalies
        A = (ens .- x_mean) / sqrt(N_ens - 1.0)

        # step 2b: compute the SVD for the two-sided projected model error covariance
        F = svd(A)
        Σ_inv = Diagonal([1.0 ./ F.S[1:N_ens-1]; 0.0]) 
        p_inv = F.V * Σ_inv * transpose(F.U)
        G = Symmetric(1.0I + (N_ens - 1.0) * p_inv * conditioning * transpose(p_inv))
        
        # step 2c: compute the model error adjusted anomalies
        A = A * square_root(G)

        # step 3: compute the ensemble in observation space
        Z = H * ens

        # step 4: compute the ensemble mean in observation space
        y_mean = mean(Z, dims=2)
        
        # step 5: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = inv(square_root(obs_cov))
        
        # then compute the weighted anomalies
        S = (Z .- y_mean) / sqrt(N_ens - 1.0)
        S = obs_sqrt_inv * S

        # step 6: compute the weighted innovation
        δ = obs_sqrt_inv * ( obs - y_mean )
       
        # step 7: compute the transform matrix
        T = inv(Symmetric(1.0I + transpose(S) * S))
        
        # step 8: compute the analysis weights
        w = T * transpose(S) * δ

        # step 9: compute the square root of the transform
        T_sqrt = sqrt(T)
        
        # step 10:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

        # step 11: package the transform output tuple
        T_sqrt, w, U

    elseif analysis=="etks_adaptive"
        ## This computes the transform of the ETKF update as in Asch, Bocquet, Nodet
        # but using a computation of the contribution of the model error covariance matrix Q
        # in the square root as in Raanes et al. 2015 and the adaptive inflation from the
        # frequentist estimator for the model error covariance
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the ensemble mean
        x_mean = mean(ens, dims=2)

        # step 2a: compute the normalized anomalies
        A = (ens .- x_mean) / sqrt(N_ens - 1.0)

        if !(m_err[1] == Inf)
            # step 2b: compute the SVD for the two-sided projected model error covariance
            F_ens = svd(A)
            mean_err = mean(m_err, dims=2)
            A_err = (m_err .- mean_err) / sqrt(length(mean_err) - 1.0)
            F_err = svd(A_err)
            Σ_pinv = Diagonal([1.0 ./ F_ens.S[1:N_ens-1]; 0.0]) 

            # step 2c: compute the square root covariance with model error anomaly contribution
            # in the ensemble space dimension, note the difference in equation due to the normalized
            # anomalies
            G = Symmetric(I +  Σ_pinv * transpose(F_ens.U) * F_err.U *
                          Diagonal(F_err.S.^2) * transpose(F_err.U) * 
                          F_ens.U * Σ_pinv)
            
            G = F_ens.V * square_root(G) * F_ens.Vt

            # step 2c: compute the model error adjusted anomalies
            A = A * G

        end

        # step 3: compute the ensemble in observation space
        Z = H * ens

        # step 4: compute the ensemble mean in observation space
        y_mean = mean(Z, dims=2)
        
        # step 5: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = inv(square_root(obs_cov))
        
        # then compute the weighted anomalies
        S = (Z .- y_mean) / sqrt(N_ens - 1.0)
        S = obs_sqrt_inv * S

        # step 6: compute the weighted innovation
        δ = obs_sqrt_inv * ( obs - y_mean )
       
        # step 7: compute the transform matrix
        T = inv(Symmetric(1.0I + transpose(S) * S))
        
        # step 8: compute the analysis weights
        w = T * transpose(S) * δ

        # step 9: compute the square root of the transform
        T_sqrt = sqrt(T)
        
        # step 10:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

        # step 11: package the transform output tuple
        T_sqrt, w, U

    elseif analysis=="enkf-n" || analysis=="enks-n"
        ## This computes the dual form of the EnKF-N transform as in bocquet & raanes 2015
        ## NOTE: may want to try a higher order hessian-based approach later, for now this simply uses
        ## the Brent method for the argmin problem
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the ensemble mean
        x_mean = mean(ens, dims=2)

        # step 2: compute the non-normalized anomalies
        A = ens .- x_mean
        
        # step 3: compute the ensemble mean and anomalies in observation space
        y_mean = H * x_mean
        Y = H * A

        # step 4: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = inv(square_root(obs_cov))
        
        # then compute the weighted anomalies
        S = obs_sqrt_inv * Y

        # step 5: compute the weighted innovation
        δ = obs_sqrt_inv * ( obs - y_mean )
        
        # step 6: compute the SVD for the simplified cost function, gauge weights and range
        F = svd(S)
        ϵ_N = 1.0 + (1.0 / N_ens)
        ζ_l = 0.000001
        ζ_u = (N_ens + 1.0) / ϵ_N
        
        # step 7: define the dual cost function derived in singular value form
        function D(ζ)
            cost = I - (F.U * Diagonal( F.S.^2.0 ./ (ζ .+ F.S.^2.0) ) * transpose(F.U) )
            cost = transpose(δ) * cost * δ .+ ϵ_N * ζ .+ (N_ens + 1.0) * log((N_ens + 1.0) / ζ) .- (N_ens + 1.0)
            cost[1]
        end
        
        ## The below is defined for possible Hessian-based minimization 
        ## NOTE: we get failure to converge in test cases with Optim library
        #
        #function D_v(ζ)
        #    ζ = ζ[1]
        #    cost = I - (F.U * Diagonal( F.S.^2.0 ./ (ζ .+ F.S.^2.0) ) * transpose(F.U) )
        #    cost = transpose(δ) * cost * δ .+ ϵ_N * ζ .+ (N_ens + 1.0) * log((N_ens + 1.0) / ζ) .- (N_ens + 1.0)
        #    cost[1]
        #end

        #function g!(grad, ζ)
        #    ζ = ζ[1]
        #    grad = transpose(δ) * F.U * Diagonal( - F.S.^2.0 .* (ζ .+ F.S.^2.0).^(-2.0) ) * transpose(F.U) * δ
        #    grad = grad .+ ϵ_N  .- (N_ens + 1.0) / ζ
        #end

        #function h!(hess, ζ)
        #    ζ = ζ[1]
        #    hess = transpose(δ) * F.U * Diagonal( 2.0 * F.S.^2.0 .* (ζ .+ F.S.^2.0).^(-3.0) ) * transpose(F.U) * δ
        #    hess = hess .+ (N_ens + 1.0) * ζ^(-2.0)
        #end

        #lx = [ζ_l]
        #ux = [ζ_u]
        #ζ_0 = [(ζ_u + ζ_l)/2.0]
        #df = TwiceDifferentiable(D_v, g!, h!, ζ_0)
        #dfc = TwiceDifferentiableConstraints(lx, ux)
        #ζ_b = optimize(D_v, h!, g!, ζ_0, x_abstol =10e-12)


        # step 8: find the argmin
        ζ_a = optimize(D, ζ_l, ζ_u)
        diag_vals = ζ_a.minimizer .+ F.S.^2.0

        # step 9: compute the update weights
        # NOTE: for consistency with the ETKF update code, we scale to account 
        # for the normalized anomalies in the update step
        w = F.V * Diagonal( F.S ./ diag_vals ) * transpose(F.U) * δ * sqrt(N_ens - 1.0)

        # step 10: compute the update transform
        # NOTE: for consistency with the ETKF update code, we scale to account 
        # for the normalized anomalies in the update step
        H_sqrt_inv = Symmetric(Diagonal( F.S ./ diag_vals) * transpose(F.U) * δ * 
                               transpose(δ) * F.U * Diagonal( F.S ./ diag_vals))
        H_sqrt_inv = Diagonal(diag_vals) - ( (2.0 * ζ_a.minimizer^2.0) / (N_ens + 1.0) ) * H_sqrt_inv
        H_sqrt_inv = Symmetric(F.V * inv(square_root(H_sqrt_inv)) * F.Vt * sqrt(N_ens - 1.0))
        
        # step 11:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

        # step 12: package the transform output tuple
        H_sqrt_inv, w, U
    
    elseif analysis=="ienks-bundle" || analysis=="ienks-transform"
        # this computes the weighted observed anomalies as per the Gauss-Newton, bundle or transform
        # version of the IEnKS -- bundle uses a small uniform scalar epsilon, transform uses a matrix
        # as the conditioning, with bundle used by default
        # this returns a sequential-in-time value for the cost function gradient and hessian
        
        # step 1: compute the ensemble mean in observation space
        y_mean = mean(H * ens, dims=2)
        
        # step 2: compute the observed anomalies, inversely proportional to the conditioning matrix
        y_anom = (H * ens .- y_mean) * inv(conditioning)

        # step 3: compute the cost function gradient term
        inv_obs_cov = inv(obs_cov)
        ∇J = transpose(y_anom) * inv_obs_cov * (obs - y_mean)

        # step 4: compute the cost function gradient term
        hess_J = transpose(y_anom) * inv_obs_cov * y_anom

        # return tuple of the gradient and hessian terms
        ∇J, hess_J

    end
end


########################################################################################################################
# auxilliary function for updating stochastic/ deterministic transform ensemble kalman filter 

function ens_update!(ens::Array{Float64,2}, transform::T0) where {T0 <: TransM}

    if T0 <: Array{Float64,2}
        # step 1: update the ensemble with right transform
        ens * transform 
    
    else
        # step 0: infer dimensions and unpack the transform
        sys_dim, N_ens = size(ens)
        T_sqrt, w, U = transform
        
        # step 1: compute the ensemble mean
        X_mean = mean(ens, dims=2)

        # step 2: compute the normalized anomalies
        A = (ens .- X_mean) / sqrt(N_ens - 1.0)

        # step 3: compute the update
        ens_transform = w .+ T_sqrt * U * sqrt(N_ens - 1.0)
        X_mean .+ A * ens_transform
    end
end


########################################################################################################################
# general filter code 

function ensemble_filter(analysis::String, ens::Array{Float64,2}, H::T1, obs::Vector{Float64}, 
                         obs_cov::T2, state_infl::Float64, kwargs::Dict{String,Any}) where {T1 <: ObsH, T2 <: CovM}

    """General filter analysis step

    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
    case, a value for the parameter covariance inflation should be included in addition to the state covariance
    inflation."""

    # step 0: infer the system, observation and ensemble dimensions 
    sys_dim, N_ens = size(ens)
    obs_dim = length(obs)

    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]
        param_infl = kwargs["param_infl"]

    else
        state_dim = sys_dim
    end

    # step 1: compute the tranform and update ensemble
    ens = ens_update!(ens, transform(analysis, ens, H, obs, obs_cov)) 

    # step 2a: compute multiplicative inflation of state variables
    ens = inflate_state!(ens, state_infl, sys_dim, state_dim)

    # step 2b: if including an extended state of parameter values,
    # compute multiplicative inflation of parameter values
    if state_dim != sys_dim
        ens = inflate_param!(ens, param_infl, sys_dim, state_dim)
    end

    Dict{String,Array{Float64,2}}("ens" => ens)
end


########################################################################################################################
# classical version lag_shift_smoother

function ls_smoother_classic(analysis::String, ens::Array{Float64,2}, H::T1, obs::Array{Float64,2}, 
                             obs_cov::T2, state_infl::Float64, kwargs::Dict{String,Any}) where {T1 <: ObsH, T2 <: CovM}

    """Lag-shift ensemble kalman smoother analysis step, classical version

    This version of the lag-shift enks uses the last filtered state for the forecast, differentiated from the hybrid
    and iterative schemes which will use the once or multiple-times re-analized posterior for the initial condition
    for the forecast of the states to the next shift.

    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
    case, a value for the parameter covariance inflation should be included in addition to the state covariance
    inflation."""
    
    # step 0: unpack kwargs, posterior contains length lag past states ending with ens as final entry
    f_steps = kwargs["f_steps"]::Int64
    step_model = kwargs["step_model"]
    posterior = kwargs["posterior"]::Array{Float64,3}
    
    # infer the ensemble, obs, and system dimensions, observation sequence includes shift forward times
    obs_dim, shift = size(obs)
    sys_dim, N_ens, lag = size(posterior)

    # optional parameter estimation
    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]::Int64
        param_infl = kwargs["param_infl"]::Float64
        param_wlk = kwargs["param_wlk"]::Float64

    else
        state_dim = sys_dim
    end

    # step 1: create storage for the forecast and filter values over the DAW
    forecast = Array{Float64}(undef, sys_dim, N_ens, shift)
    filtered = Array{Float64}(undef, sys_dim, N_ens, shift)

    # step 2: forward propagate the ensemble and analyze the observations
    for s in 1:shift
        # initialize posterior for the special case lag=shift
        if lag==shift
            posterior[:, :, s] = ens
        end
        
        # step 2a: propagate between observation times
        for j in 1:N_ens
            for k in 1:f_steps
                ens[:, j] = step_model(ens[:, j], kwargs, 0.0)
            end
        end

        # step 2b: store the forecast to compute ensemble statistics before observations become available
        forecast[:, :, s] = ens

        # step 2c: perform the filtering step
        trans = transform(analysis, ens, H, obs[:, s], obs_cov)
        ens = ens_update!(ens, trans)

        # compute multiplicative inflation of state variables
        ens = inflate_state!(ens, state_infl, sys_dim, state_dim)

        # if including an extended state of parameter values,
        # compute multiplicative inflation of parameter values
        if state_dim != sys_dim
            ens = inflate_param!(ens, param_infl, sys_dim, state_dim)
        end

        # store the filtered states
        filtered[:, :, s] = ens
        
        # step 2e: re-analyze the posterior in the lag window of states
        for l in 1:lag
            posterior[:, :, l] = ens_update!(posterior[:, :, l], trans)
        end
    end
            
    # step 3: if performing parameter estimation, apply the parameter model
    if state_dim != sys_dim
        param_ens = ens[state_dim + 1:end , :]
        param_ens = param_ens + param_wlk * rand(Normal(), size(param_ens))
        ens[state_dim + 1:end, :] = param_ens
    end
    
    Dict{String,Array{Float64}}(
                                "ens" => ens, 
                                "post" =>  posterior, 
                                "fore" => forecast, 
                                "filt" => filtered
                               ) 
end

#########################################################################################################################
# single iteration, correlation-based lag_shift_smoother

function ls_smoother_hybrid(analysis::String, ens::Array{Float64,2}, H::T1, obs::Array{Float64,2}, 
                             obs_cov::T2, state_infl::Float64, kwargs::Dict{String,Any}) where {T1 <: ObsH, T2 <: CovM}

    """Lag-shift ensemble kalman smoother analysis step, hybrid version

    This version of the lag-shift enks uses the final re-analyzed posterior initial state for the forecast, 
    which is pushed forward in time from the initial conidtion to shift-number of observation times.

    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
    case, a value for the parameter covariance inflation should be included in addition to the state covariance
    inflation. If the analysis method is 'etks_adaptive', this utilizes the past analysis means to construct an 
    innovation-based estimator for the model error covariances.  This is formed by the expectation step in the
    expectation maximization algorithm dicussed by Tandeo et al. 2021."""
    
    # step 0: unpack kwargs, posterior contains length lag past states ending with ens as final entry
    f_steps = kwargs["f_steps"]::Int64
    step_model = kwargs["step_model"]
    posterior = kwargs["posterior"]::Array{Float64,3}
    
    # for the adaptive inflation shceme
    if analysis == "etks_adaptive"
        # analysis_innovations will contain the sequence of the last cycle's analysis mean 
        # states over the current DAW and the innovations computed in the previous DAW-shift times
        analysis_innovations = kwargs["analysis"]::Array{Float64,2}
    end
    
    # infer the ensemble, obs, and system dimensions, observation sequence includes lag forward times
    obs_dim, lag = size(obs)
    sys_dim, N_ens, shift = size(posterior)

    # optional parameter estimation
    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]::Int64
        param_infl = kwargs["param_infl"]::Float64
        param_wlk = kwargs["param_wlk"]::Float64

    else
        state_dim = sys_dim
    end

    # make a copy of the intial ens for re-analysis
    ens_0 = copy(ens)
    
    # spin to be used on the first lag-assimilations -- this makes the smoothed time-zero re-analized prior
    # the first initial condition for the future iterations regardless of sda or mda settings
    spin = kwargs["spin"]::Bool
    
    # step 1: create storage for the posterior, forecast and filter values over the DAW
    # only the shift-last and shift-first values are stored as these represent the newly forecasted values and
    # last-iterate posterior estimate respectively
    if spin
        forecast = Array{Float64}(undef, sys_dim, N_ens, lag)
        filtered = Array{Float64}(undef, sys_dim, N_ens, lag)
    else
        forecast = Array{Float64}(undef, sys_dim, N_ens, shift)
        filtered = Array{Float64}(undef, sys_dim, N_ens, shift)
    end
    
    if analysis == "etks_adaptive"
        # creat storage for the analysis means computed at each forward step of the current DAW
        analysis_means = Array{Float64}(undef, sys_dim, lag)
    end
    
    # multiple data assimilation (mda) is optional, read as boolean variable
    mda = kwargs["mda"]::Bool
    
    if mda
        # set the observation and re-balancing weights
        reb_weights = kwargs["reb_weights"]::Vector{Float64}
        obs_weights = kwargs["obs_weights"]::Vector{Float64}

        # set iteration count for the initial rebalancing step followed by mda
        i = 0
        
        # the posterior statistics are computed in the zeroth pass with rebalancing
        posterior[:, :, 1] = ens_0
        
        # we make a single iteration with SDA, with MDA we make a rebalancing step on the zeroth iteration
        while i <=1 
            # step 2: forward propagate the ensemble and analyze the observations
            for l in 1:lag
                # step 2a: propagate between observation times
                for j in 1:N_ens
                    for k in 1:f_steps
                        ens[:, j] = step_model(ens[:, j], kwargs, 0.0)
                    end
                end
                if i == 0
                    # step 2b: store the forecast to compute ensemble statistics before observations become available
                    # for MDA, this is on the zeroth iteration through the DAW
                    if spin
                        # store all new forecast states
                        forecast[:, :, l] = ens
                    elseif (l > (lag - shift))
                        # only store forecasted states for beyond unobserved times beyond previous forecast windows
                        forecast[:, :, l - (lag - shift)] = ens
                    end
                    
                    # step 2c: perform the filtering step with rebalancing weights 
                    trans = transform(analysis, ens, H, obs[:, l], obs_cov * reb_weights[l])
                    ens = ens_update!(ens, trans)

                    if spin 
                        # compute multiplicative inflation of state variables
                        ens = inflate_state!(ens, state_infl, sys_dim, state_dim)

                        # if including an extended state of parameter values,
                        # compute multiplicative inflation of parameter values
                        if state_dim != sys_dim
                            ens = inflate_param!(ens, param_infl, sys_dim, state_dim)
                        end
                        
                        # store all new filtered states
                        filtered[:, :, l] = ens
                    
                    elseif l > (lag - shift)
                        # store the filtered states for previously unobserved times, not mda values
                        filtered[:, :, l - (lag - shift)] = ens
                    end
                    
                    # step 2d: compute the re-analyzed initial condition for posterior statistics with rebalancing step
                    posterior[:, :, 1] = ens_update!(posterior[:, :, 1], trans)
                    
                    # on final iteration, inflate the covariance
                    if l == lag
                        # compute multiplicative inflation of state variables
                        posterior[:, :, 1] = inflate_state!(posterior[:, :, 1], state_infl, sys_dim, state_dim)

                        # if including an extended state of parameter values,
                        # compute multiplicative inflation of parameter values
                        if state_dim != sys_dim
                            posterior[:, :, 1] = inflate_param!(posterior[:, :, 1], param_infl, sys_dim, state_dim)
                        end
                    end
                else
                    # step 2c: perform the filtering step with mda weights
                    trans = transform(analysis, ens, H, obs[:, l], obs_cov * obs_weights[l])
                    ens = ens_update!(ens, trans)
                    
                    # re-analyzed initial conditions are computed in the mda step
                    ens_0 = ens_update!(ens_0, trans)
                end
            end
            # reset the ensemble with the prior for mda and step forward the iteration count,
            ens = copy(ens_0)
            i+=1
        end
    else
        # step 2: forward propagate the ensemble and analyze the observations
        for l in 1:lag
            # step 2a: propagate between observation times
            for j in 1:N_ens
                for k in 1:f_steps
                    ens[:, j] = step_model(ens[:, j], kwargs, 0.0)
                end
            end
            if spin
                # step 2b: store the forecast to compute ensemble statistics before observations become available
                # if spin, store all new forecast states
                forecast[:, :, l] = ens
                
                # step 2c: apply the transformation and update step
                trans = transform(analysis, ens, H, obs[:, l], obs_cov)
                ens = ens_update!(ens, trans)
                
                # compute multiplicative inflation of state variables
                ens = inflate_state!(ens, state_infl, sys_dim, state_dim)

                # if including an extended state of parameter values,
                # compute multiplicative inflation of parameter values
                if state_dim != sys_dim
                    ens = inflate_param!(ens, param_infl, sys_dim, state_dim)
                end
                
                # store all new filtered states
                filtered[:, :, l] = ens
            
                if analysis == "etks_adaptive"
                    # store the analysis means for future statistics
                    analysis_means[:, l] = mean(ens, dims=2)
                end

                # step 2d: compute the re-analyzed initial condition if we have an assimilation update
                ens_0 = ens_update!(ens_0, trans)
            
            elseif l > (lag - shift)
                # step 2b: store the forecast to compute ensemble statistics before observations become available
                # if not spin, only store forecasted states for beyond unobserved times beyond previous forecast windows
                forecast[:, :, l - (lag - shift)] = ens
                
                # step 2c: apply the transformation and update step
                if analysis == "etks_adaptive"
                    trans = transform(analysis, ens, H, obs[:, l], obs_cov, 
                                      m_err=analysis_innovations)
                else
                    trans = transform(analysis, ens, H, obs[:, l], obs_cov)
                end

                ens = ens_update!(ens, trans)
                
                # store the filtered states for previously unobserved times, not mda values
                filtered[:, :, l - (lag - shift)] = ens
                
                if analysis == "etks_adaptive"
                    # store the analysis means for future statistics
                    analysis_means[:, l] = mean(ens, dims=2)
                end

                # step 2d: compute the re-analyzed initial condition if we have an assimilation update
                ens_0 = ens_update!(ens_0, trans)
            elseif analysis == "etks_adaptive"
                # store the analysis means for future statistics
                analysis_means[:, l] = mean(ens, dims=2)

                # compute the innovation versus the last cycle's analysis state
                analysis_innovations[:, l + shift] = analysis_innovations[:, l + shift] - analysis_means[:, l]
            end
        end
        # reset the ensemble with the re-analyzed prior 
        ens = copy(ens_0)

        if analysis == "etks_adaptive"
            # reset the analysis innovations for the next DAW
            analysis_innovations = copy(analysis_means)
        end
    end

    # step 3: propagate the posterior initial condition forward to the shift-forward time
    # step 3a: inflate the posterior covariance
    ens = inflate_state!(ens, state_infl, sys_dim, state_dim)
    
    # if including an extended state of parameter values,
    # compute multiplicative inflation of parameter values
    if state_dim != sys_dim
        ens = inflate_param!(ens, param_infl, sys_dim, state_dim)
    end

    # step 3b: if performing parameter estimation, apply the parameter model
    if state_dim != sys_dim
        param_ens = ens[state_dim + 1:end , :]
        param_ens = param_ens + param_wlk * rand(Normal(), size(param_ens))
        ens[state_dim + 1:end, :] = param_ens
    end

    # step 3c: propagate the re-analyzed, resampled-in-parameter-space ensemble up by shift
    # observation times
    for s in 1:shift
        if !mda
            posterior[:, :, s] = ens
        end
        for j in 1:N_ens
            for k in 1:f_steps
                ens[:, j] = step_model(ens[:, j], kwargs, 0.0)
            end
        end
        if analysis == "etks_adaptive"
            # compute the analysis innovations over the shift states for the next DAW
            analysis_innovations[:, s] = analysis_innovations[:, s] - mean(ens, dims=2)
        end
    end

    if analysis == "etks_adaptive"
       return  Dict{String,Array{Float64}}(
                                           "ens" => ens, 
                                           "post" =>  posterior, 
                                           "fore" => forecast, 
                                           "filt" => filtered,
                                           "anal" => analysis_innovations
                                          )
    else
       return  Dict{String,Array{Float64}}(
                                           "ens" => ens, 
                                           "post" =>  posterior, 
                                           "fore" => forecast, 
                                           "filt" => filtered,
                                          ) 
    end
end


#########################################################################################################################

function ls_smoother_iterative(analysis::String, ens::Array{Float64,2}, H::T1, obs::Array{Float64,2}, 
                             obs_cov::T2, state_infl::Float64, kwargs::Dict{String,Any};
                             ϵ::Float64=0.0001, tol::Float64=0.001, max_iter::Int64=50) where {T1 <: ObsH, T2 <: CovM}


    """Lag-shift ensemble IEnKS analysis step, algorithm 4, Bocquet & Sakov 2014

    This version of the lag-shift ienks uses the final re-analyzed posterior initial state for the forecast, 
    which is pushed forward in time from the initial conidtion to shift-number of observation times.

    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
    case, a value for the parameter covariance inflation should be included in addition to the state covariance
    inflation."""
    
    # step 0: unpack kwargs, posterior contains length lag past states ending with ens as final entry
    f_steps = kwargs["f_steps"]::Int64
    step_model = kwargs["step_model"]
    posterior = kwargs["posterior"]::Array{Float64,3}
    
    # infer the ensemble, obs, and system dimensions, observation sequence includes lag forward times
    obs_dim, lag = size(obs)
    sys_dim, N_ens, shift = size(posterior)

    # optional parameter estimation
    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]::Int64
        param_infl = kwargs["param_infl"]::Float64
        param_wlk = kwargs["param_wlk"]::Float64

    else
        state_dim = sys_dim
    end

    # spin to be used on the first lag-assimilations -- this makes the smoothed time-zero re-analized prior
    # the first initial condition for the future iterations regardless of sda or mda settings
    spin = kwargs["spin"]::Bool
    
    # multiple data assimilation (mda) is optional, read as boolean variable
    mda = kwargs["mda"]::Bool
    if mda
        obs_weights = kwargs["obs_weights"]::Vector{Float64}
    else
        obs_weights = ones(lag)
    end

    # step 1: define the data assimilation quantities

    # step 1a: create storage for the posterior, forecast and filter values over the DAW
    # only the shift-last and shift-first values are stored as these represent the newly forecasted values and
    # last-iterate posterior estimate respectively
    if spin
        forecast = Array{Float64}(undef, sys_dim, N_ens, lag)
        filtered = Array{Float64}(undef, sys_dim, N_ens, lag)
    else
        forecast = Array{Float64}(undef, sys_dim, N_ens, shift)
        filtered = Array{Float64}(undef, sys_dim, N_ens, shift)
    end

    # step 1b: define the initial correction and iteration count
    w = zeros(N_ens)
    i = 0

    # step 1c: compute the initial ensemble mean and normalized anomalies, storage for the
    # sequentially computed iterated mean, gradient and and hessian terms 
    ens_mean_0 = mean(ens, dims=2)
    ens_mean_iter = copy(ens_mean_0) 
    anom_0 = ens .- ens_mean_0 

    if spin || mda
        ∇J = Array{Float64}(undef, N_ens, lag)
        hess_J = Array{Float64}(undef, N_ens, N_ens, lag)
    else
        ∇J = Array{Float64}(undef, N_ens, shift)
        hess_J = Array{Float64}(undef, N_ens, N_ens, shift)
    end

    # pre-allocate these variables as global for the loops
    hessian = Symmetric(Array{Float64}(undef, N_ens, N_ens))
    new_ens = Array{Float64}(undef, sys_dim, N_ens)

    # step 1e: define the conditioning for bundle versus transform varaints
    if analysis == "ienks-bundle"
        T = ϵ*I
    elseif analysis == "ienks-transform"
        T = 1.0*I
    end

    # step 2: begin iterative optimization
    while i <= max_iter 
        # step 2a: redefine the conditioned ensemble with updated mean, after first forecast
        if i > 0
            ens = ens_mean_iter .+ anom_0 * T
        end

        # step 2b: forward propagate the ensemble and sequentially store the forecast or construct cost function
        for l in 1:lag
            # propagate between observation times
            for j in 1:N_ens
                for k in 1:f_steps
                    ens[:, j] = step_model(ens[:, j], kwargs, 0.0)
                end
            end

            # step 2c: store the forecast to compute ensemble statistics before observations become available
            # NOTE: this should only occur on the first pass before observations are assimilated into the first prior
            # and performed with the un-scaled or conditioned ensemble
            if i == 0
                if spin
                    forecast[:, :, l] = ens
                elseif l > (lag - shift)
                    forecast[:, :, l - (lag - shift)] = ens
                end

            # step 2d: compute the sequential terms of the gradient and hessian of the cost function if in spin, 
            # multiple DA (mda=true) or whenever the lag-forecast steps take us to new observations (l>(lag - shift))
            # after the initial forecast
            elseif mda || spin
                ∇J[:,l], hess_J[:, :, l] = transform(analysis, ens, H, obs[:, l], obs_cov * obs_weights[l], conditioning=T)

            elseif l > (lag - shift)
                ∇J[:,l - (lag - shift)], 
                hess_J[:, :, l - (lag - shift)] = transform(analysis, ens, H, obs[:, l], obs_cov * obs_weights[l], conditioning=T)

            end

        end

        if i > 0
            # step 2e: formally compute the gradient and the hessian from the sequential components, 
            # perform Gauss-Newton step after forecast iteration
            gradient = (N_ens - 1.0) * w - sum(∇J, dims=2)
            hessian = Symmetric((N_ens - 1.0) * I + dropdims(sum(hess_J, dims=3), dims=3))

            if analysis == "ienks-transform"
                T = inv(square_root(hessian)) 
            end

            Δw = hessian \ gradient
            w -= Δw 
            # step 2f: update the mean via the increment, always with the zeroth iterate of the ensemble,
            # but store the next iterate of the ensemble for reuse in the final analysis
            ens_mean_iter = ens_mean_0 + anom_0 * w
            
            if norm(Δw) < tol
                break
            end
        end
        
        # update the iteration count
        i+=1
    end
                
    # step 3: compute posterior initial condiiton and propagate forward in time
    
    # step 3a: perform the analysis of the ensemble
    if analysis == "ienks-transform"
        H_minus_half = T
    else
        H_minus_half = inv(square_root(hessian))
    end
    
    # we compute the analyzed ensemble by the iterated mean and the transformed original anomalies
    U = rand_orth(N_ens)
    ens = ens_mean_iter .+ sqrt(N_ens - 1.0) * anom_0 * H_minus_half * U

    # step 3b: if performing parameter estimation, apply the parameter model
    if state_dim != sys_dim
        param_ens = ens[state_dim + 1:end , :]
        param_ens = param_ens + param_wlk * rand(Normal(), size(param_ens))
        ens[state_dim + 1:end, :] = param_ens
    end

    # step 3c: propagate the re-analyzed, resampled-in-parameter-space ensemble up by shift
    # observation times, store the filtered state as the forward propagated value at new observation
    # times, store the posterior at the times discarded at the next shift
    for l in 1:lag
        if l <= shift
            posterior[:, :, l] = ens
        end
        for j in 1:N_ens
            for k in 1:f_steps
                ens[:, j] = step_model(ens[:, j], kwargs, 0.0)
            end
        end
        if l == shift
            new_ens = copy(ens)
        end
        if spin
            filtered[:, :, l] = ens

        elseif l > lag - shift
            filtered[:, :, l-(lag - shift)] = ens
        end
    end
    
    # store and inflate the forward posterior at the new initial condition
    ens = new_ens
    ens = inflate_state!(ens, state_infl, sys_dim, state_dim)

    # if including an extended state of parameter values,
    # compute multiplicative inflation of parameter values
    if state_dim != sys_dim
        ens = inflate_param!(ens, param_infl, sys_dim, state_dim)
    end

    Dict{String,Array{Float64}}(
                                "ens" => ens, 
                                "post" =>  posterior, 
                                "fore" => forecast, 
                                "filt" => filtered,
                                "iterations" => Array{Float64}([i])
                               ) 
end


#########################################################################################################################

end

#########################################################################################################################
## IEnKF-T-LM
#
#
#def ietlm(X_ext_ens, H, obs, obs_cov, f_steps, f, h, tau=0.001, e1=0,
#         inflation=1.0, tol=0.001, l_max=40):
#
#    """This produces an analysis ensemble via transform as in algorithm 3, bocquet sakov 2012"""
#
#    # step 0: infer the ensemble, obs, and state dimensions
#    [sys_dim, N_ens] = np.shape(X_ext_ens)
#    obs_dim = len(obs)
#
#    # step 1: we compute the ensemble mean and non-normalized anomalies
#    X_mean_0 = np.mean(X_ext_ens, axis=1)
#    A_t = X_ext_ens.transpose() - X_mean_0
#
#    # step 2: we define the initial iterative minimization parameters
#    l = 0
#    nu = 2
#    w = np.zeros(N_ens)
#    
#    # step 3: update the mean via the w increment
#    X_mean_1 = X_mean_0 + A_t.transpose() @ w
#    X_mean_tmp = copy.copy(X_mean_1)
#
#    # step 4: evolve the ensemble mean forward in time, and transform into observation space
#    for k in range(f_steps):
#        # propagate ensemble mean one step forward
#        X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)
#
#    # define the observed mean by the propagated mean in the observation space
#    Y_mean = H @ X_mean_tmp
#
#    # step 5: Define the initial transform
#    T = np.eye(N_ens)
#    
#    # step 6: redefine the ensemble with the updated mean and the transform
#    X_ext_ens = (X_mean_1 + T @ A_t).transpose()
#
#    # step 7: loop over the discretization steps between observations to produce a forecast ensemble
#    for k in range(f_steps):
#        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
#
#    # step 8: compute the forecast anomalies in the observation space, via the observed, evolved mean and the 
#    # observed, forward ensemble, conditioned by the transform
#    Y_ens = H @ X_ext_ens
#    Y_ens_t = np.linalg.inv(T).transpose() @ (Y_ens.transpose() - Y_mean) 
#
#    # step 9: compute the cost function in ensemble space
#    J = 0.5 * (obs - Y_mean) @ np.linalg.inv(obs_cov) @ (obs - Y_mean) + 0.5 * (N_ens - 1) * w @ w
#    
#    # step 10: compute the approximate gradient of the cost function
#    grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)
#
#    # step 11: compute the approximate hessian of the cost function
#    hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()
#
#    # step 12: compute the infinity norm of the jacobian and the max of the hessian diagonal
#    flag = np.max(np.abs(grad_J)) > e1
#    mu = tau * np.max(np.diag(hess))
#
#    # step 13: while loop
#    while flag: 
#        if l > l_max:
#            break
#
#        # step 14: set the iteration count forward
#        l+= 1
#        
#        # step 15: solve the system for the w increment update
#        δ_w = solve(hess + mu * np.eye(N_ens),  -1 * grad_J)
#
#        # step 16: check if the increment is sufficiently small to terminate
#        if np.sqrt(δ_w @ δ_w) < tol:
#            # step 17: flag false to terminate
#            flag = False
#
#        # step 18: begin else
#        else:
#            # step 19: reset the ensemble adjustment
#            w_prime = w + δ_w
#            
#            # step 20: reset the initial ensemble with the new adjustment term
#            X_mean_1 = X_mean_0 + A_t.transpose() @ w_prime
#            
#            # step 21: forward propagate the new ensemble mean, and transform into observation space
#            X_mean_tmp = copy.copy(X_mean_1)
#            for k in range(f_steps):
#                X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)
#            
#            Y_mean = H @ X_mean_tmp
#
#            # steps 22 - 24: define the parameters for the confidence region
#            L = 0.5 * δ_w @ (mu * δ_w - grad_J)
#            J_prime = 0.5 * (obs - Y_mean) @ np.linalg.inv(obs_cov) @ (obs - Y_mean) + 0.5 * (N_ens -1) * w_prime @ w_prime
#            theta = (J - J_prime) / L
#
#            # step 25: evaluate if new correction needed
#            if theta > 0:
#                
#                # steps 26 - 28: update the cost function, the increment, and the past ensemble, conditioned with the
#                # transform
#                J = J_prime
#                w = w_prime
#                X_ext_ens = (X_mean_1 + T.transpose() @ A_t).transpose()
#
#                # step 29: integrate the ensemble forward in time
#                for k in range(f_steps):
#                    X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
#
#                # step 30: compute the forward anomlaies in the observation space, by the forward evolved mean and forward evolved
#                # ensemble
#                Y_ens = H @ X_ext_ens
#                Y_ens_t = np.linalg.inv(T).transpose() @ (Y_ens.transpose() - Y_mean)
#
#                # step 31: compute the approximate gradient of the cost function
#                grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)
#
#                # step 32: compute the approximate hessian of the cost function
#                hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()
#
#                # step 33: define the transform as the inverse square root of the hessian
#                V, Sigma, V_t = np.linalg.svd(hess)
#                T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t
#
#                # steps 34 - 35: compute the tolerance and correction parameters
#                flag = np.max(np.abs(grad_J)) > e1
#                mu = mu * np.max([1/3, 1 - (2 * theta - 1)**3])
#                nu = 2
#
#            # steps 36 - 37: else statement, update mu and nu
#            else:
#                mu = mu * nu
#                nu = nu * 2
#
#            # step 38: end if
#        # step 39: end if
#    # step 40: end while
#
#    # step 41: perform update to the initial mean with the new defined anomaly transform 
#    X_mean_1 = X_mean_0 + A_t.transpose() @ w
#
#    # step 42: define the transform as the inverse square root of the hessian, bundle version only
#    #V, Sigma, V_t = np.linalg.svd(hess)
#    #T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t
#
#    # step 43: compute the updated ensemble by the transform conditioned anomalies and updated mean
#    X_ext_ens = (T.transpose() @ A_t + X_mean_1).transpose()
#    
#    # step 44: forward propagate the ensemble to the observation time 
#    for k in range(f_steps):
#        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
#   
#    # step 45: compute the ensemble with inflation
#    X_mean_2 = np.mean(X_ext_ens, axis=1)
#    A_t = X_ext_ens.transpose() - X_mean_2
#    infl = np.eye(N_ens) * inflation
#    X_ext_ens = (X_mean_2 + infl @  A_t).transpose()
#
#    return X_ext_ens
#
#########################################################################################################################
## IEnKF-B-LM
#
#
#def ieblm(X_ext_ens, H, obs, obs_cov, f_steps, f, h, tau=0.001, e1=0, epsilon=0.0001,
#         inflation=1.0, tol=0.001, l_max=40):
#
#    """This produces an analysis ensemble as in algorithm 3, bocquet sakov 2012"""
#
#    # step 0: infer the ensemble, obs, and state dimensions
#    [sys_dim, N_ens] = np.shape(X_ext_ens)
#    obs_dim = len(obs)
#
#    # step 1: we compute the ensemble mean and non-normalized anomalies
#    X_mean_0 = np.mean(X_ext_ens, axis=1)
#    A_t = X_ext_ens.transpose() - X_mean_0
#
#    # step 2: we define the initial iterative minimization parameters
#    l = 0
#    
#    # NOTE: MARC'S VERSION HAS NU SET TO ONE FIRST AND THEN ITERATES ON THIS IN PRODUCTS
#    # OF TWO    
#    #nu = 2
#    nu = 1
#
#    w = np.zeros(N_ens)
#    
#    # step 3: update the mean via the w increment
#    X_mean_1 = X_mean_0 + A_t.transpose() @ w
#    X_mean_tmp = copy.copy(X_mean_1)
#
#    # step 4: evolve the ensemble mean forward in time, and transform into observation space
#    for k in range(f_steps):
#        X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)
#
#    Y_mean = H @ X_mean_tmp
#
#    # step 5: Define the initial transform, transform version only
#    # T = np.eye(N_ens)
#    
#    # step 6: redefine the ensemble with the updated mean, rescaling by epsilon
#    X_ext_ens = (X_mean_1 + epsilon * A_t).transpose()
#
#    # step 7: loop over the discretization steps between observations to produce a forecast ensemble
#    for k in range(f_steps):
#        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
#
#    # step 8: compute the anomalies in the observation space, via the observed, evolved mean and the observed, 
#    # forward ensemble, rescaling by epsilon
#    Y_ens = H @ X_ext_ens
#    Y_ens_t = (Y_ens.transpose() - Y_mean) / epsilon
#
#    # step 9: compute the cost function in ensemble space
#    J = 0.5 * (obs - Y_mean) @ np.linalg.inv(obs_cov) @ (obs - Y_mean) + 0.5 * (N_ens - 1) * w @ w
#    
#    # step 10: compute the approximate gradient of the cost function
#    grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)
#
#    # step 11: compute the approximate hessian of the cost function
#    hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()
#
#    # step 12: compute the infinity norm of the jacobian and the max of the hessian diagonal
#    # NOTE: MARC'S VERSION DOES NOT HAVE A FLAG BASED ON THE INFINITY NORM OF THE GRADIENT
#    # THIS IS ALSO PROBABLY A TRIVIAL FLAG
#    # flag = np.max(np.abs(grad_J)) > e1
#    
#    # NOTE: MARC'S FLAG
#    flag = True
#
#    # NOTE: MARC'S VERSION USES MU=1 IN THE FIRST ITERATION AND NEVER MAKES
#    # THIS DECLARATION IN TERMS OF TAU AND HESS
#    # mu = tau * np.max(np.diag(hess))
#    mu = 1
#    
#    # step 13: while loop
#    while flag: 
#        if l > l_max:
#            print(l)
#            break
#
#        # step 14: set the iteration count forward
#        l+= 1
#        
#        # NOTE: MARC'S RE-DEFINITION OF MU AND NU
#        mu *= nu
#        nu *= 2
#
#        # step 15: solve the system for the w increment update
#        δ_w = solve(hess + mu * np.eye(N_ens),  -1 * grad_J)
#
#        # step 16: check if the increment is sufficiently small to terminate
#        # NOTE: MARC'S VERSION NORMALIZES THE LENGTH RELATIVE TO THE ENSEMBLE SIZE
#        if np.sqrt(δ_w @ δ_w) < tol:
#            # step 17: flag false to terminate
#            flag = False
#            print(l)
#
#        # step 18: begin else
#        else:
#            # step 19: reset the ensemble adjustment
#            w_prime = w + δ_w
#            
#            # step 20: reset the initial ensemble with the new adjustment term
#            X_mean_1 = X_mean_0 + A_t.transpose() @ w_prime
#            
#            # step 21: forward propagate the new ensemble mean, and transform into observation space
#            X_mean_tmp = copy.copy(X_mean_1)
#            for k in range(f_steps):
#                X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)
#            
#            Y_mean = H @ X_mean_tmp
#
#            # steps 22 - 24: define the parameters for the confidence region
#            L = 0.5 * δ_w @ (mu * δ_w - grad_J)
#            J_prime = 0.5 * (obs - Y_mean) @ np.linalg.inv(obs_cov) @ (obs - Y_mean) + 0.5 * (N_ens -1) * w_prime @ w_prime
#            theta = (J - J_prime) / L
#            
#            # step 25: evaluate if new correction needed
#            if theta > 0:
#                
#                # steps 26 - 28: update the cost function, the increment, and the past ensemble, rescaled with epsilon
#                J = J_prime
#                w = w_prime
#                X_ext_ens = (X_mean_1 + epsilon * A_t).transpose()
#
#                # step 29: integrate the ensemble forward in time
#                for k in range(f_steps):
#                    X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
#
#                # step 30: compute the forward anomlaies in the observation space, by the forward evolved mean and forward evolved
#                # ensemble
#                Y_ens = H @ X_ext_ens
#                Y_ens_t = (Y_ens.transpose() - Y_mean) / epsilon
#
#                # step 31: compute the approximate gradient of the cost function
#                grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)
#
#                # step 32: compute the approximate hessian of the cost function
#                hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()
#
#                # step 33: define the transform as the inverse square root of the hessian, transform version only
#                #V, Sigma, V_t = np.linalg.svd(hess)
#                #T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t
#
#                # steps 34 - 35: compute the tolerance and correction parameters
#                # NOTE: TRIVIAL FLAG?
#                # flag = np.max(np.abs(grad_J)) > e1
#                
#                mu = mu * np.max([1/3, 1 - (2 * theta - 1)**3])
#                
#                # NOTE: ADJUSTMENT HERE TO MATCH NU TO MARC'S CODE
#                # nu = 2
#                nu = 1
#
#            # steps 36 - 37: else statement, update mu and nu
#            #else:
#            #    mu = mu * nu
#            #    nu = nu * 2
#
#            # step 38: end if
#        # step 39: end if
#    # step 40: end while
#    
#    # step 41: perform update to the initial mean with the new defined anomaly transform 
#    X_mean_1 = X_mean_0 + A_t.transpose() @ w
#
#    # step 42: define the transform as the inverse square root of the hessian
#    V, Sigma, V_t = np.linalg.svd(hess)
#    T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t
#
#    # step 43: compute the updated ensemble by the transform conditioned anomalies and updated mean
#    X_ext_ens = (T.transpose() @ A_t + X_mean_1).transpose()
#    
#    # step 44: forward propagate the ensemble to the observation time 
#    for k in range(f_steps):
#        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
#    
#    # step 45: compute the ensemble with inflation
#    X_mean_2 = np.mean(X_ext_ens, axis=1)
#    A_t = X_ext_ens.transpose() - X_mean_2
#    infl = np.eye(N_ens) * inflation
#    X_ext_ens = (X_mean_2 + infl @  A_t).transpose()
#
#    return X_ext_ens
#
