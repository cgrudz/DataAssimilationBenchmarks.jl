##############################################################################################
module EnsembleKalmanSchemes
##############################################################################################
# imports and exports
using Random, Distributions, Statistics
using LinearAlgebra, SparseArrays
using ..DataAssimilationBenchmarks
using Optim, LineSearches
export alternating_obs_operator, analyze_ens, analyze_ens_param, rand_orth, 
       inflate_state!, inflate_param!, transform_R, ens_gauss_newton, square_root,
       square_root_inv, ensemble_filter, ls_smoother_classic,
       ls_smoother_single_iteration, ls_smoother_gauss_newton
##############################################################################################
# Main methods, debugged and validated
##############################################################################################
"""
    alternating_projector(x::VecA(T), obs_dim::Int64) where T <: Real

Utility method produces a projection of alternating vector components via slicing.
```
return x
```

This operator takes a single model state `x` of type [`VecA`](@ref) and maps this data to
alternating entries.  The operator selects components of the vector
based on the observation dimension.  States correpsonding to even state dimension indices
are removed from the state vector until the observation dimension is appropriate.
If the observation dimension is less than half the state dimension, states corresponding
to odd state dimension idices are subsequently removed until the observation dimension
is appropriate.
"""
function alternating_projector(x::VecA(T), obs_dim::Int64) where T <: Real
    sys_dim = length(x)
    if obs_dim == sys_dim

    elseif (obs_dim / sys_dim) > 0.5
        # the observation dimension is greater than half the state dimension, so we
        # remove only the trailing odd-index rows equal to the difference
        # of the state and observation dimension
        R = sys_dim - obs_dim
        indx = 1:(sys_dim - 2 * R)
        indx = [indx; sys_dim - 2 * R + 2: 2: sys_dim]
        x = x[indx]

    elseif (obs_dim / sys_dim) == 0.5
        # the observation dimension is equal to half the state dimension so we remove exactly
        # half the rows, corresponding to those with even-index
        x = x[1:2:sys_dim, :]

    else
        # the observation dimension is less than half of the state dimension so that we
        # remove all even rows and then all but the remaining, leading obs_dim rows
        x = x[1:2:sys_dim]
        x = x[1:obs_dim]
    end
    return x
end


##############################################################################################
"""
    alternating_projector(ens::ArView(T), obs_dim::Int64) where T <: Real

Utility method produces a projection of alternating ensemble components in-place via slicing.
```
return ens
```

This operator takes either a truth twin time series or an ensemble of states of type
[`ArView`](@ref), and maps this data to alternating row components.  The truth twin in
this version is assumed to be 2D, where the first index corresponds to the state dimension
and the second index corresponds to the time dimension.  The ensemble is assumed to be
2D where the first index corresponds to the state dimension and the second index
corresponds to the ensemble dimension.
States correpsonding to even state dimension indices are removed from the state
vector until the observation dimension is appropriate.  If the observation dimension is
less than half the state dimension, states corresponding to odd state dimension idices
are subsequently removed until the observation dimension is appropriate.
"""
function alternating_projector(ens::ArView(T), obs_dim::Int64) where T <: Real
    sys_dim, N_ens = size(ens)
    if obs_dim == sys_dim

    elseif (obs_dim / sys_dim) > 0.5
        # the observation dimension is greater than half the state dimension, so we
        # remove only the trailing odd-index rows equal to the difference
        # of the state and observation dimension
        R = sys_dim - obs_dim
        indx = 1:(sys_dim - 2 * R)
        indx = [indx; sys_dim - 2 * R + 2: 2: sys_dim]
        ens = ens[indx, :]

    elseif (obs_dim / sys_dim) == 0.5
        # the observation dimension is equal to half the state dimension so we remove exactly
        # half the rows, corresponding to those with even-index
        ens = ens[1:2:sys_dim, :]

    else
        # the observation dimension is less than half of the state dimension so that we
        # remove all even rows and then all but the remaining, leading obs_dim rows
        ens = ens[1:2:sys_dim, :]
        ens = ens[1:obs_dim, :]
    end
    return ens
end


##############################################################################################
"""
    alternating_obs_operator(x::VecA(T), obs_dim::Int64, kwargs::StepKwargs) where T <: Real

This produces observations of alternating state vector components for generating pseudo-data.
```
return obs
```

This operator takes a single model state `x` of type [`VecA`](@ref) and maps this data to
the observation space via the method [`alternating_projector`](@ref) and (possibly) a 
nonlinear transform.
The `γ` parameter (optional) in `kwargs` of type  [`StepKwargs`](@ref) controls the
component-wise transformation of the remaining state vector components mapped to the
observation space.  For `γ=1.0`, there is no transformation applied, and the observation
operator acts as a linear projection onto the remaining components of the state vector,
equivalent to not specifying `γ`. For `γ>1.0`, the nonlinear observation operator of 
[Asch, et al. (2016).](https://epubs.siam.org/doi/book/10.1137/1.9781611974546),
pg. 181 is applied, which limits to the identity for `γ=1.0`.  If `γ=0.0`, the quadratic
observation operator of [Hoteit, et al. (2012).](https://journals.ametsoc.org/view/journals/mwre/140/2/2011mwr3640.1.xml)
is applied to the remaining state components.  If `γ<0.0`, the exponential observation
operator of [Wu, et al. (2014).](https://npg.copernicus.org/articles/21/955/2014/)
is applied to the remaining state vector components.
"""
function alternating_obs_operator(x::VecA(T), obs_dim::Int64,
                                  kwargs::StepKwargs) where T <: Real
    sys_dim = length(x)
    if haskey(kwargs, "state_dim")
        # performing parameter estimation, load the dynamic state dimension
        state_dim = kwargs["state_dim"]::Int64
        
        # observation operator for extended state, without observing extended state components
        obs = copy(x[1:state_dim])
        
        # proceed with alternating observations of the regular state vector
        sys_dim = state_dim
    else
        obs = copy(x)
    end

    # project the state vector into the correct components
    obs = alternating_projector(obs, obs_dim)

    if haskey(kwargs, "γ")
        γ = kwargs["γ"]::Float64
        if γ > 1.0
            obs .= (obs / 2.0) .* ( 1.0 .+ ( abs.(obs) / 10.0 ).^(γ - 1.0) )

        elseif γ == 0.0
            obs .= 0.05*obs.^2.0

        elseif γ < 0.0
            obs .= obs .* exp.(-γ * obs)
        end
    end
    return obs
end


##############################################################################################
"""
    alternating_obs_operator(ens::ArView(T), obs_dim::Int64,
                             kwargs::StepKwargs) where T <: Real

This produces observations of alternating state vector components for generating pseudo-data.
```
return obs
```

This operator takes either a truth twin time series or an ensemble of states of type
[`ArView`](@ref), and maps this data to the observation space via the method
[`alternating_projector`](@ref) and (possibly) a nonlinear transform.  The truth twin in
this version is assumed to be 2D, where the first index corresponds to the state dimension
and the second index corresponds to the time dimension.  The ensemble is assumed to be
2D where the first index corresponds to the state dimension and the second index
corresponds to the ensemble dimension.
"""
function alternating_obs_operator(ens::ArView(T), obs_dim::Int64,
                                  kwargs::StepKwargs) where T <: Real
    sys_dim, N_ens = size(ens)

    if haskey(kwargs, "state_dim")
        # performing parameter estimation, load the dynamic state dimension
        state_dim = kwargs["state_dim"]::Int64
        
        # observation operator for extended state, without observing extended state components
        obs = copy(ens[1:state_dim, :])
        
        # proceed with alternating observations of the regular state vector
        sys_dim = state_dim
    else
        obs = copy(ens)
    end

    # project the state vector into the correct components
    obs = alternating_projector(obs, obs_dim)

    if haskey(kwargs, "γ")
        γ = kwargs["γ"]::Float64
        if γ > 1.0
            for i in 1:N_ens
                x = obs[:, i]
                obs[:, i] .= (x / 2.0) .* ( 1.0 .+ ( abs.(x) / 10.0 ).^(γ - 1.0) )
            end

        elseif γ == 0.0
            obs = 0.05*obs.^2.0

        elseif γ < 0.0
            for i in 1:N_ens
                x = obs[:, i]
                obs[:, i] .= x .* exp.(-γ * x)
            end
        end
    end
    return obs
end


##############################################################################################
"""
    analyze_ens(ens::ArView(T), truth::VecA(T)) where T <: Float64  

Computes the ensemble state RMSE as compared with truth twin, and the ensemble spread.
```
return rmse, spread
```

Note: the ensemble `ens` should only include the state vector components to compare with the
truth twin state vector `truth`, without replicates of the model parameters.  These can be
passed as an [`ArView`](@ref) for efficient memory usage.
"""
function analyze_ens(ens::ArView(T), truth::VecA(T)) where T <: Float64

    # infer the shapes
    sys_dim, N_ens = size(ens)

    # compute the ensemble mean
    x_bar = mean(ens, dims=2)

    # compute the RMSE of the ensemble mean
    rmse = sqrt(mean( (truth - x_bar).^2.0))

    # compute the spread as in whitaker & louge 98 by the standard deviation 
    # of the mean square deviation of the ensemble from its mean
    spread = sqrt( ( 1.0 / (N_ens - 1.0) ) * sum(mean((ens .- x_bar).^2.0, dims=1)))

    # return the tuple pair
    rmse, spread
end


##############################################################################################
"""
    analyze_ens_param(ens::ArView(T), truth::VecA(T)) where T <: Float64

Computes the ensemble parameter RMSE as compared with truth twin, and the ensemble spread.
```
return rmse, spread
```

Note: the ensemble `ens` should only include the extended state vector components
consisting of model parameter replicates to compare with the truth twin's governing
model parameters `truth`.  These can be passed as an [`ArView`](@ref) for
efficient memory usage.
"""
function analyze_ens_param(ens::ArView(T), truth::VecA(T)) where T <: Float64
    # infer the shapes
    param_dim, N_ens = size(ens)

    # compute the ensemble mean
    x_bar = mean(ens, dims=2)

    # compute the RMSE of relative to the magnitude of the parameter
    rmse = sqrt( mean( (truth - x_bar).^2.0 ./ truth.^2.0 ) )

    # compute the spread as in whitaker & louge 98 by the standard deviation
    # of the mean square deviation of the ensemble from its mean,
    # with the weight by the size of the parameter square
    spread = sqrt( ( 1.0 / (N_ens - 1.0) ) * 
                   sum(mean( (ens .- x_bar).^2.0 ./ 
                             (ones(param_dim, N_ens) .* truth.^2.0), dims=1)))
    
    # return the tuple pair
    rmse, spread
end


##############################################################################################
"""
    rand_orth(N_ens::Int64)

This generates a random, mean-preserving, orthogonal matrix as in [Sakov & Oke
2008](https://journals.ametsoc.org/view/journals/mwre/136/3/2007mwr2021.1.xml), depending on
the esemble size `N_ens`.
```
return U
```
"""
function rand_orth(N_ens::Int64)
    # generate the random, mean preserving orthogonal transformation within the 
    # basis given by the B matrix
    Q = rand(Normal(), N_ens - 1, N_ens - 1)
    Q, R = qr!(Q)
    U_p =  zeros(N_ens, N_ens)
    U_p[1, 1] = 1.0
    U_p[2:end, 2:end] = Q

    # generate the B basis for which the first basis vector is the vector of 1/sqrt(N)
    b_1 = ones(N_ens) / sqrt(N_ens)
    B = zeros(N_ens, N_ens)
    B[:, 1] = b_1

    # note, this uses the "full" QR decomposition so that the singularity is encoded in R
    # and B is a full-size orthogonal matrix
    B, R = qr!(B)
    U = B * U_p * transpose(B) 
    U
end


##############################################################################################
"""
    inflate_state!(ens::ArView(T), inflation::Float64, sys_dim::Int64,
                   state_dim::Int64) where T <: Float64 

Applies multiplicative covariance inflation to the state components of the ensemble matrix.
```
return ens
```

The first index of the ensemble matrix `ens` corresponds to the length `sys_dim` (extended)
state dimension while the second index corresponds to the ensemble dimension.  Dynamic state
variables are assumed to be in the leading `state_dim` rows of `ens`, while extended state
parameter replicates are after. Multiplicative inflation is performed only in the leading
components of the ensemble anomalies from the ensemble mean, in-place in memory.
"""
function inflate_state!(ens::ArView(T), inflation::Float64, sys_dim::Int64,
                        state_dim::Int64) where T <: Float64 
    if inflation != 1.0
        x_mean = mean(ens, dims=2)
        X = ens .- x_mean
        infl =  Matrix(1.0I, sys_dim, sys_dim) 
        infl[1:state_dim, 1:state_dim] .*= inflation 
        ens .= x_mean .+ infl * X
        return ens
    end
end


##############################################################################################
"""
    inflate_param!(ens::ArView(T), inflation::Float64, sys_dim::Int64,
                   state_dim::Int64) where T <: Float64 

Applies multiplicative covariance inflation to parameter replicates in the ensemble matrix.
```
return ens
```

The first index of the ensemble matrix `ens` corresponds to the length `sys_dim` (extended)
state dimension while the second index corresponds to the ensemble dimension.  Dynamic state
variables are assumed to be in the leading `state_dim` rows of `ens`, while extended state
parameter replicates are after. Multiplicative inflation is performed only in the trailing 
`state_dim + 1: state_dim` components of the ensemble anomalies from the ensemble mean,
in-place in memory.
"""
function inflate_param!(ens::ArView(T), inflation::Float64, sys_dim::Int64,
                        state_dim::Int64) where T <: Float64 
    if inflation == 1.0
        return ens
    else
        x_mean = mean(ens, dims=2)
        X = ens .- x_mean
        infl =  Matrix(1.0I, sys_dim, sys_dim) 
        infl[state_dim+1: end, state_dim+1: end] .*= inflation
        ens .= x_mean .+ infl * X
        return ens
    end
end


##############################################################################################
"""
    square_root(M::CovM(T)) where T <: Real 

Computes the square root of covariance matrices with parametric type.

Multiple dispatches for the method are defined according to the sub-type of [`CovM`](@ref),
where the square roots of `UniformScaling` and `Diagonal` covariance matrices are computed
directly, while the square roots of  the more general class of `Symmetric` covariance
matrices are computed via the singular value decomposition, for stability and accuracy
for close-to-singular matrices.

```
return S
```
"""
function square_root(M::UniformScaling{T}) where T <: Real 
    S = M^0.5
end

function square_root(M::Diagonal{T, Vector{T}}) where T <: Real 
    S = sqrt(M)
end

function square_root(M::Symmetric{T, Matrix{T}}) where T <: Real 
    F = svd(M)
    S = Symmetric(F.U * Diagonal(sqrt.(F.S)) * F.Vt)
end

##############################################################################################
"""
    square_root_inv(M::CovM(T); sq_rt::Bool=false, inverse::Bool=false,
                    full::Bool=false) where T <: Real 

Computes the square root inverse of covariance matrices with parametric type.

Multiple dispatches for the method are defined according to the sub-type of [`CovM`](@ref),
where the square root inverses of `UniformScaling` and `Diagonal` covariance matrices
are computed directly, while the square root inverses of the more general class of
`Symmetric` covariance matrices are computed via the singular value decomposition, for
stability and accuracy for close-to-singular matrices. This will optionally return a
computation of the inverse and the square root itself all as a byproduct of the singular
value decomposition for efficient numerical computation of ensemble
analysis / update routines. 

Optional keyword arguments are specified as:
 * `sq_rt=true` returns the matrix square root in addition to the square root inverse
 * `inverse=true` returns the matrix inverse in addition to the square root inverse
 * `full=true` returns the square root and the matrix inverse in addition to the square
    root inverse
and are evaluated in the above order.

Output follows control flow:
```
if sq_rt
    return S_inv, S
elseif inverse
    return S_inv, M_inv
elseif full
    return S_inv, S, M_inv
else
    return S_inv
end
```
"""
function square_root_inv(M::UniformScaling{T}; sq_rt::Bool=false, inverse::Bool=false,
                         full::Bool=false) where T <: Real 
        if sq_rt
            S = M^0.5
            S_inv = S^(-1.0)
            S_inv, S
        elseif inverse
            M_inv = M^(-1.0)
            S_inv = M_inv^0.5
            S_inv, M_inv
        elseif full
            M_inv = M^(-1.0)
            S = M^0.5
            S_inv = S^(-1.0)
            S_inv, S, M_inv
        else
            S_inv = M^(-0.5)
            S_inv
        end
end

function square_root_inv(M::Diagonal{T, Vector{T}}; sq_rt::Bool=false, inverse::Bool=false,
                         full::Bool=false) where T <: Real 
    if sq_rt
        S = sqrt(M)
        S_inv = inv(S)
        S_inv, S
    
    elseif inverse 
        M_inv = inv(M)
        S_inv = sqrt(M_inv)
        S_inv, M_inv
    
    elseif full
        S = sqrt(M)
        S_inv = inv(S)
        M_inv = inv(M)
        S_inv, S, M
    else
        S_inv = M.^(-0.5)
        S_inv
    end
end

function square_root_inv(M::Symmetric{T, Matrix{T}}; sq_rt::Bool=false, inverse::Bool=false,
                         full::Bool=false) where T <: Real 
    # stable square root inverse for close-to-singular inverse calculations
    F = svd(M)
    if sq_rt 
        # take advantage of the SVD calculation to produce both the square root inverse
        # and square root simultaneously
        S_inv = Symmetric(F.U * Diagonal(1.0 ./ sqrt.(F.S)) * F.Vt) 
        S = Symmetric(F.U * Diagonal(sqrt.(F.S)) * F.Vt) 
        S_inv, S
    elseif inverse
        # take advantage of the SVD calculation to produce the square root inverse
        # and inverse calculations all at once
        S_inv = Symmetric(F.U * Diagonal(1.0 ./ sqrt.(F.S)) * F.Vt)
        M_inv = Symmetric(F.U * Diagonal(1.0 ./ F.S) * F.Vt)
        S_inv, M_inv
    elseif full
        # take advantage of the SVD calculation to produce the square root inverse,
        # square root and inverse calculations all at once
        S_inv = Symmetric(F.U * Diagonal(1.0 ./ sqrt.(F.S)) * F.Vt)
        S = Symmetric(F.U * Diagonal(sqrt.(F.S)) * F.Vt)
        M_inv = Symmetric(F.U * Diagonal(1.0 ./ F.S) * F.Vt)
        S_inv, S, M_inv
    else
        # only return the square root inverse, if other calculations are not necessary
        S_inv = Symmetric(F.U * Diagonal(1.0 ./ sqrt.(F.S)) * F.Vt)
        S_inv
    end
end
##############################################################################################
"""
    transform_R(analysis::String, ens::ArView(T), obs::VecA(T),        
                obs_cov::CovM(T), kwargs::StepKwargs; conditioning::ConM=1000.0I, 
                m_err::ArView(T)=(1.0 ./ zeros(1,1)),
                tol::Float64 = 0.0001,
                j_max::Int64=40,
                Q::CovM(T)=1.0I) where T <: Float64

Computes the ensemble transform and related values for various flavors of ensemble
Kalman schemes. The output type is a tuple containing a right transform of the ensemble
anomalies, the weights for the mean update and a random orthogonal transformation
for stabilization:
```
return trans, w, U
```
where the tuple is of type [`TransM`](@ref).
`m_err`, `tol`, `j_max`, `Q` are optional arguments depending on the `analysis`, with 
default values provided.

Serves as an auxilliary function for EnKF, ETKF(-N), EnKS, ETKS(-N), where
"analysis" is a string which determines the type of transform update.  The observation
error covariance `obs_cov` is of type [`CovM`](@ref), the conditioning matrix `conditioning`
is of type [`ConM`](@ref), the keyword arguments dictionary `kwargs` is of type
[`StepKwargs`](@ref) and the model error covariance matrix `Q` is of type [`CovM`](@ref).

Currently validated `analysis` options:
 * `analysis=="etkf" || analysis=="etks"` computes the deterministic ensemble transform 
   as in the ETKF described in [Grudzien, et al.
   2021](https://gmd.copernicus.org/preprints/gmd-2021-306/).
 * `analysis[1:7]=="mlef-ls" || analysis[1:7]=="mles-ls"` computes the maximum likelihood
   ensemble filter transform described in [Grudzien, et al. 
   2021](https://gmd.copernicus.org/preprints/gmd-2021-306/), optimizing the nonlinear
   cost function with Newton-based 
   [line searches](https://julianlsolvers.github.io/LineSearches.jl/stable/).
 * `analysis[1:4]=="mlef" || analysis[1:4]=="mles"` computes the maximum likelihood     
   ensemble filter transform described in
   [Grudzien, et al. 2021](https://gmd.copernicus.org/preprints/gmd-2021-306/),
   optimizing the nonlinear
   cost function with simple Newton-based scheme. 
 * `analysis=="enkf-n-dual" || analysis=="enks-n-dual"` 
   computes the dual form of the EnKF-N transform as in [Bocquet, et al.
   2015](https://npg.copernicus.org/articles/22/645/2015/)
   Note: this cannot be used with the nonlinear observation operator.
   This uses the Brent method for the argmin problem as this
   has been more reliable at finding a global minimum than Newton optimization.
 * `analysis=="enkf-n-primal" || analysis=="enks-n-primal"`
   computes the primal form of the EnKF-N transform as in [Bocquet, et al.
   2015](https://npg.copernicus.org/articles/22/645/2015/),
   [Grudzien, et al. 2021](https://gmd.copernicus.org/preprints/gmd-2021-306/).
   This differs from the MLEF/S-N in that there is no approximate linearization of
   the observation operator in the EnKF-N, this only handles the approximation error
   with respect to the adaptive inflation. This uses a simple Newton-based
   minimization of the cost function for the adaptive inflation.
 * `analysis=="enkf-n-primal-ls" || analysis=="enks-n-primal-ls"`
   computes the primal form of the EnKF-N transform as in [Bocquet, et al.
   2015](https://npg.copernicus.org/articles/22/645/2015/),
   [Grudzien, et al. 2021](https://gmd.copernicus.org/preprints/gmd-2021-306/).
   This differs from the MLEF/S-N in that there is no approximate linearization of
   the observation operator in the EnKF-N, this only handles the approximation error
   with respect to the adaptive inflation. This uses a Newton-based
   minimization of the cost function for the adaptive inflation with
   [line searches](https://julianlsolvers.github.io/LineSearches.jl/stable/).
"""
function transform_R(analysis::String, ens::ArView(T), obs::VecA(T), 
                     obs_cov::CovM(T), kwargs::StepKwargs; conditioning::ConM(T)=1000.0I, 
                     m_err::ArView(T)=(1.0 ./ zeros(1,1)),
                     tol::Float64 = 0.0001,
                     j_max::Int64=40,
                     Q::CovM(T)=1.0I) where T <: Float64 

    if analysis=="etkf" || analysis=="etks"
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the ensemble in observation space
        Y = alternating_obs_operator(ens, obs_dim, kwargs)

        # step 2: compute the ensemble mean in observation space
        y_mean = mean(Y, dims=2)
        
        # step 3: compute the sensitivity matrix in observation space
        obs_sqrt_inv = square_root_inv(obs_cov)
        S = obs_sqrt_inv * (Y .- y_mean )

        # step 4: compute the weighted innovation
        δ = obs_sqrt_inv * ( obs - y_mean )
       
        # step 5: compute the approximate hessian
        hessian = Symmetric((N_ens - 1.0)*I + transpose(S) * S)
        
        # step 6: compute the transform matrix, transform matrix inverse and
        # hessian inverse simultaneously via the SVD for stability
        trans, hessian_inv = square_root_inv(hessian, inverse=true)
        
        # step 7: compute the analysis weights
        w = hessian_inv * transpose(S) * δ

        # step 8: generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

    elseif analysis[1:7]=="mlef-ls" || analysis[1:7]=="mles-ls"
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)
        
        # step 1: set up inputs for the optimization 
        
        # step 1a: inial choice is no change to the mean state
        ens_mean_0 = mean(ens, dims=2)
        anom_0 = ens .- ens_mean_0
        w = zeros(N_ens)

        # step 1b: pre-compute the observation error covariance square root
        obs_sqrt_inv = square_root_inv(obs_cov)

        # step 1c: define the conditioning and parameters for finite size formalism if needed
        if analysis[end-5:end] == "bundle"
            trans = inv(conditioning) 
            trans_inv = conditioning
        elseif analysis[end-8:end] == "transform"
            trans = Symmetric(Matrix(1.0*I, N_ens, N_ens))
            trans_inv = Symmetric(Matrix(1.0*I, N_ens, N_ens))
        end

        if analysis[8:9] == "-n"
            # define the epsilon scaling and the effective ensemble size if finite size form
            ϵ_N = 1.0 + (1.0 / N_ens)
            N_effective = N_ens + 1.0
        end

        # step 1d: define the storage of the gradient and Hessian as global to the functions
        grad_w = Vector{Float64}(undef, N_ens)
        hess_w = Array{Float64}(undef, N_ens, N_ens)
        cost_w = 0.0

        # step 2: define the cost / gradient / hessian function to avoid repeated computations
        function fgh!(G, H, C, trans::ConM(T1), trans_inv::ConM(T1),
                      w::Vector{T1}) where T1 <: Float64
            # step 2a: define the linearization of the observation operator 
            ens_mean_iter = ens_mean_0 + anom_0 * w
            ens = ens_mean_iter .+ anom_0 * trans 
            Y = alternating_obs_operator(ens, obs_dim, kwargs)
            y_mean = mean(Y, dims=2)

            # step 2b: compute the weighted anomalies in observation space, conditioned
            # with trans inverse
            S = obs_sqrt_inv * (Y .- y_mean) * trans_inv 

            # step 2c: compute the weighted innovation
            δ = obs_sqrt_inv * (obs - y_mean)
        
            # step 2d: gradient, hessian and cost function definitions
            if G != nothing
                if analysis[8:9] == "-n"
                    ζ = 1.0 / (ϵ_N + sum(w.^2.0))
                    G[:] = N_effective * ζ * w - transpose(S) * δ
                else
                    G[:] = (N_ens - 1.0)  * w - transpose(S) * δ
                end
            end
            if H != nothing
                if analysis[8:9] == "-n"
                    H .= Symmetric((N_effective - 1.0)*I + transpose(S) * S)
                else
                    H .= Symmetric((N_ens - 1.0)*I + transpose(S) * S)
                end
            end
            if C != nothing
                if analysis[8:9] == "-n"
                    y_mean_iter = alternating_obs_operator(ens_mean_iter, obs_dim, kwargs)
                    δ = obs_sqrt_inv * (obs - y_mean_iter)
                    return N_effective * log(ϵ_N + sum(w.^2.0)) + sum(δ.^2.0)
                else
                    y_mean_iter = alternating_obs_operator(ens_mean_iter, obs_dim, kwargs)
                    δ = obs_sqrt_inv * (obs - y_mean_iter)
                    return (N_ens - 1.0) * sum(w.^2.0) + sum(δ.^2.0)
                end
            end
            nothing
        end
        function newton_ls!(grad_w, hess_w, trans::ConM(T1), trans_inv::ConM(T1),
                            w::Vector{T1}, linesearch) where T1 <: Float64
            # step 2e: find the Newton direction and the transform update if needed
            fx = fgh!(grad_w, hess_w, cost_w, trans, trans_inv, w)
            p = -hess_w \ grad_w
            if analysis[end-8:end] == "transform"
                trans_tmp, trans_inv_tmp = square_root_inv(Symmetric(hess_w), sq_rt=true)
                trans .= trans_tmp
                trans_inv .= trans_inv_tmp
            end
            
            # step 2f: univariate line search functions
            ϕ(α) = fgh!(nothing, nothing, cost_w, trans, trans_inv, w .+ α.*p)
            function dϕ(α)
                fgh!(grad_w, nothing, nothing, trans, trans_inv, w .+ α.*p)
                return dot(grad_w, p)
            end
            function ϕdϕ(α)
                phi = fgh!(grad_w, nothing, cost_w, trans, trans_inv, w .+ α.*p)
                dphi = dot(grad_w, p)
                return (phi, dphi)
            end

            # step 2g: define the linesearch
            dϕ_0 = dot(p, grad_w)
            α, fx = linesearch(ϕ, dϕ, ϕdϕ,  1.0, fx, dϕ_0)
            Δw = α * p
            w .= w + Δw
            
            return Δw
        end
        
        # step 3: optimize
        # step 3a: perform the optimization by Newton with linesearch
        # we use StrongWolfe for RMSE performance as the default linesearch
        #ln_search = HagerZhang()
        ln_search = StrongWolfe()
        j = 0
        Δw = ones(N_ens)

        while j < j_max && norm(Δw) > tol
            Δw = newton_ls!(grad_w, hess_w, trans, trans_inv, w, ln_search)
        end
        
        if analysis[8:9] == "-n"
            # peform a final inflation with the finite size cost function
            ζ = 1.0 / (ϵ_N + sum(w.^2.0))
            hess_w = ζ * I - 2.0 * ζ^2.0 * w * transpose(w) 
            hess_w = Symmetric(transpose(S) * S + (N_ens + 1.0) * hess_w)
            trans = square_root_inv(hess_w)
        
        elseif analysis[end-5:end] == "bundle"
            trans = square_root_inv(hess_w)
        end

        # step 3b:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

    elseif analysis[1:4]=="mlef" || analysis[1:4]=="mles"
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)
        
        # step 1: set up the optimization, inial choice is no change to the mean state
        ens_mean_0 = mean(ens, dims=2)
        anom_0 = ens .- ens_mean_0
        w = zeros(N_ens)

        # pre-compute the observation error covariance square root
        obs_sqrt_inv = square_root_inv(obs_cov)

        # define these variables as global compared to the while loop
        grad_w = Vector{Float64}(undef, N_ens)
        hess_w = Array{Float64}(undef, N_ens, N_ens)
        S = Array{Float64}(undef, obs_dim, N_ens)
        ens_mean_iter = copy(ens_mean_0)

        # define the conditioning 
        if analysis[end-5:end] == "bundle"
            trans = inv(conditioning) 
            trans_inv = conditioning
        elseif analysis[end-8:end] == "transform"
            trans = 1.0*I
            trans_inv = 1.0*I
        end
        
        # step 2: perform the optimization by simple Newton
        j = 0
        if analysis[5:6] == "-n"
            # define the epsilon scaling and the effective ensemble size if finite size form
            ϵ_N = 1.0 + (1.0 / N_ens)
            N_effective = N_ens + 1.0
        end
        
        while j < j_max
            # step 2a: compute the observed ensemble and ensemble mean 
            ens_mean_iter = ens_mean_0 + anom_0 * w
            ens = ens_mean_iter .+ anom_0 * trans
            Y = alternating_obs_operator(ens, obs_dim, kwargs)
            y_mean = mean(Y, dims=2)

            # step 2b: compute the weighted anomalies in observation space, conditioned
            # with trans inverse
            S = obs_sqrt_inv * (Y .- y_mean) * trans_inv 

            # step 2c: compute the weighted innovation
            δ = obs_sqrt_inv * (obs - y_mean)
        
            # step 2d: compute the gradient and hessian
            if analysis[5:6] == "-n" 
                # for finite formalism, we follow the IEnKS-N convention where
                # the gradient is computed with the finite-size cost function but we use the
                # usual hessian, with the effective ensemble size
                ζ = 1.0 / (ϵ_N + sum(w.^2.0))
                grad_w = N_effective * ζ * w - transpose(S) * δ
                hess_w = Symmetric((N_effective - 1.0)*I + transpose(S) * S)
            else
                grad_w = (N_ens - 1.0)  * w - transpose(S) * δ
                hess_w = Symmetric((N_ens - 1.0)*I + transpose(S) * S)
            end
            
            # step 2e: perform Newton approximation, simultaneously computing
            # the update transform trans with the SVD based inverse at once
            if analysis[end-8:end] == "transform"
                trans, trans_inv, hessian_inv = square_root_inv(hess_w, full=true)
                Δw = hessian_inv * grad_w 
            else
                Δw = hess_w \ grad_w
            end

            # 2f: update the weights
            w -= Δw 
            
            if norm(Δw) < tol
                break
            else
                # step 2g: update the iterative mean state
                j+=1
            end
        end

        if analysis[5:6] == "-n"
            # peform a final inflation with the finite size cost function
            ζ = 1.0 / (ϵ_N + sum(w.^2.0))
            hess_w = ζ * I - 2.0 * ζ^2.0 * w * transpose(w) 
            hess_w = Symmetric(transpose(S) * S + (N_ens + 1.0) * hess_w)
            trans = square_root_inv(hess_w)
        
        elseif analysis[end-5:end] == "bundle"
            trans = square_root_inv(hess_w)
        end

        # step 7:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

    elseif analysis=="etkf-sqrt-core" || analysis=="etks-sqrt-core"
        ### NOTE: STILL DEVELOPMENT CODE, NOT DEBUGGED 
        # needs to be revised for the calculation with unweighted anomalies
        # Uses the contribution of the model error covariance matrix Q
        # in the square root as in Raanes, et al. 2015
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
        ## NOTE: want to
        G = Symmetric(1.0I + (N_ens - 1.0) * p_inv * Q * transpose(p_inv))
        
        # step 2c: compute the model error adjusted anomalies
        A = A * square_root(G)

        # step 3: compute the ensemble in observation space
        Y = alternating_obs_operator(ens, obs_dim, kwargs)

        # step 4: compute the ensemble mean in observation space
        y_mean = mean(Y, dims=2)
        
        # step 5: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = square_root_inv(obs_cov)
        
        # then compute the weighted anomalies
        S = (Y .- y_mean) / sqrt(N_ens - 1.0)
        S = obs_sqrt_inv * S

        # step 6: compute the weighted innovation
        δ = obs_sqrt_inv * ( obs - y_mean )
       
        # step 7: compute the transform matrix
        trans = inv(Symmetric(1.0I + transpose(S) * S))
        
        # step 8: compute the analysis weights
        w = trans * transpose(S) * δ

        # step 9: compute the square root of the transform
        trans = sqrt(trans)
        
        # step 10:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

    elseif analysis=="enkf-n-dual" || analysis=="enks-n-dual"
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the observed ensemble and ensemble mean
        Y = alternating_obs_operator(ens, obs_dim, kwargs)
        y_mean = mean(Y, dims=2)

        # step 2: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = square_root_inv(obs_cov)
        
        # then compute the sensitivity matrix in observation space 
        S = obs_sqrt_inv * (Y .- y_mean)
 
        # step 5: compute the weighted innovation
        δ = obs_sqrt_inv * (obs - y_mean)
        
        # step 6: compute the SVD for the simplified cost function, gauge weights and range
        F = svd(S)
        ϵ_N = 1.0 + (1.0 / N_ens)
        ζ_l = 0.000001
        ζ_u = (N_ens + 1.0) / ϵ_N
        
        # step 7: define the dual cost function derived in singular value form
        function D(ζ::Float64)
            cost = I - (F.U * Diagonal( F.S.^2.0 ./ (ζ .+ F.S.^2.0) ) * transpose(F.U) )
            cost = transpose(δ) * cost * δ .+ ϵ_N * ζ .+
                   (N_ens + 1.0) * log((N_ens + 1.0) / ζ) .- (N_ens + 1.0)
            cost[1]
        end
        
        # The below is defined for possible Hessian-based minimization 
        # NOTE: standard Brent's method appears to be more reliable at finding a
        # global minimizer with some basic tests, may be tested further
        #
        #function D_v(ζ::Vector{Float64})
        #    ζ = ζ[1]
        #    cost = I - (F.U * Diagonal( F.S.^2.0 ./ (ζ .+ F.S.^2.0) ) * transpose(F.U) )
        #    cost = transpose(δ) * cost * δ .+ ϵ_N * ζ .+
        #    (N_ens + 1.0) * log((N_ens + 1.0) / ζ) .- (N_ens + 1.0)
        #    cost[1]
        #end

        #function D_prime!(storage::Vector{Float64}, ζ::Vector{Float64})
        #    ζ = ζ[1]
        #    grad = transpose(δ) * F.U * Diagonal( - F.S.^2.0 .* (ζ .+ F.S.^2.0).^(-2.0) ) *
        #           transpose(F.U) * δ
        #    storage[:, :] = grad .+ ϵ_N  .- (N_ens + 1.0) / ζ
        #end

        #function D_hess!(storage::Array{Float64}, ζ::Vector{Float64})
        #    ζ = ζ[1]
        #    hess = transpose(δ) * F.U *
        #           Diagonal( 2.0 * F.S.^2.0 .* (ζ .+ F.S.^2.0).^(-3.0) ) * transpose(F.U) * δ
        #    storage[:, :] = hess .+ (N_ens + 1.0) * ζ^(-2.0)
        #end

        #lx = [ζ_l]
        #ux = [ζ_u]
        #ζ_0 = [(ζ_u + ζ_l)/2.0]
        #df = TwiceDifferentiable(D_v, D_prime!, D_hess!, ζ_0)
        #dfc = TwiceDifferentiableConstraints(lx, ux)
        #ζ_b = optimize(D_v, D_prime!, D_hess!, ζ_0)


        # step 8: find the argmin
        ζ_a = optimize(D, ζ_l, ζ_u)
        diag_vals = ζ_a.minimizer .+ F.S.^2.0

        # step 9: compute the update weights
        w = F.V * Diagonal( F.S ./ diag_vals ) * transpose(F.U) * δ 

        # step 10: compute the update transform
        trans = Symmetric(Diagonal( F.S ./ diag_vals) * transpose(F.U) * δ * 
                               transpose(δ) * F.U * Diagonal( F.S ./ diag_vals))
        trans = Symmetric(Diagonal(diag_vals) - 
                               ( (2.0 * ζ_a.minimizer^2.0) / (N_ens + 1.0) ) * trans)
        trans = Symmetric(F.V * square_root_inv(trans) * F.Vt)
        
        # step 11:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

    elseif analysis=="enkf-n-primal" || analysis=="enks-n-primal"
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the observed ensemble and ensemble mean 
        Y = alternating_obs_operator(ens, obs_dim, kwargs)
        y_mean = mean(Y, dims=2)

        # step 2: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = square_root_inv(obs_cov)
        
        # then compute the sensitivity matrix in observation space 
        S = obs_sqrt_inv * (Y .- y_mean)

        # step 3: compute the weighted innovation
        δ = obs_sqrt_inv * (obs - y_mean)
        
        # step 4: define the epsilon scaling and the effective ensemble size
        ϵ_N = 1.0 + (1.0 / N_ens)
        N_effective = N_ens + 1.0
        
        # step 5: set up the optimization
        # step 5:a the inial choice is no change to the mean state
        w = zeros(N_ens)
        
        # step 5b: define the primal cost function
        function P(w::Vector{Float64})
            cost = (δ - S * w)
            cost = sum(cost.^2.0) + N_effective * log(ϵ_N + sum(w.^2.0))
            0.5 * cost
        end

        # step 5c: define the primal gradient
        function ∇P!(grad::Vector{Float64}, w::Vector{Float64})
            ζ = 1.0 / (ϵ_N + sum(w.^2.0))
            grad[:] = N_effective * ζ * w - transpose(S) * (δ - S * w) 
        end

        # step 5d: define the primal hessian
        function H_P!(hess::ArView(T1), w::Vector{Float64}) where T1 <: Float64
            ζ = 1.0 / (ϵ_N + sum(w.^2.0))
            hess .= ζ * I - 2.0 * ζ^2.0 * w * transpose(w) 
            hess .= transpose(S) * S + N_effective * hess
        end
        
        # step 6: perform the optimization by simple Newton
        j = 0
        trans = Array{Float64}(undef, N_ens, N_ens)
        grad_w = Array{Float64}(undef, N_ens)
        hess_w = Array{Float64}(undef, N_ens, N_ens)

        while j < j_max
            # compute the gradient and hessian
            ∇P!(grad_w, w)
            H_P!(hess_w, w)
            
            # perform Newton approximation, simultaneously computing
            # the update transform trans with the SVD based inverse at once
            trans, hessian_inv = square_root_inv(Symmetric(hess_w), inverse=true)
            Δw = hessian_inv * grad_w 
            w -= Δw 
            
            if norm(Δw) < tol
                break
            else
                j+=1
            end
        end

        # step 7:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

    elseif analysis=="enkf-n-primal-ls" || analysis=="enks-n-primal-ls"
        # step 0: infer the system, observation and ensemble dimensions 
        sys_dim, N_ens = size(ens)
        obs_dim = length(obs)

        # step 1: compute the observed ensemble and ensemble mean 
        Y = alternating_obs_operator(ens, obs_dim, kwargs)
        y_mean = mean(Y, dims=2)

        # step 2: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        obs_sqrt_inv = square_root_inv(obs_cov)
        
        # then compute the sensitivity matrix in observation space 
        S = obs_sqrt_inv * (Y .- y_mean)

        # step 3: compute the weighted innovation
        δ = obs_sqrt_inv * (obs - y_mean)
        
        # step 4: define the epsilon scaling and the effective ensemble size
        ϵ_N = 1.0 + (1.0 / N_ens)
        N_effective = N_ens + 1.0
        
        # step 5: set up the optimization
        
        # step 5:a the inial choice is no change to the mean state
        w = zeros(N_ens)
        
        # step 5b: define the primal cost function
        function J(w::Vector{Float64})
            cost = (δ - S * w)
            cost = sum(cost.^2.0) + N_effective * log(ϵ_N + sum(w.^2.0))
            0.5 * cost
        end

        # step 5c: define the primal gradient
        function ∇J!(grad::Vector{Float64}, w::Vector{Float64})
            ζ = 1.0 / (ϵ_N + sum(w.^2.0))
            grad[:] = N_effective * ζ * w - transpose(S) * (δ - S * w) 
        end

        # step 5d: define the primal hessian
        function H_J!(hess::ArView(T1), w::Vector{Float64}) where T1 <: Float64
            ζ = 1.0 / (ϵ_N + sum(w.^2.0))
            hess .= ζ * I - 2.0 * ζ^2.0 * w * transpose(w) 
            hess .= transpose(S) * S + N_effective * hess
        end
        
        # step 6: find the argmin for the update weights
        # step 6a: define the line search algorithm with Newton
        # we use StrongWolfe for RMSE performance as the default linesearch
        # method, see the LineSearches docs, alternative choice is commented below
        # ln_search = HagerZhang()
        ln_search = StrongWolfe()
        opt_alg = Newton(linesearch = ln_search)

        # step 6b: perform the optimization
        w = Optim.optimize(J, ∇J!, H_J!, w, method=opt_alg, x_tol=tol).minimizer

        # step 7: compute the update transform
        trans = Symmetric(H_J!(Array{Float64}(undef, N_ens, N_ens), w))
        trans = square_root_inv(trans)
        
        # step 8:  generate mean preserving random orthogonal matrix as in Sakov & Oke 08
        U = rand_orth(N_ens)

    end
    return trans, w, U
end

##############################################################################################
"""
    ens_gauss_newton(analysis::String, ens::ArView(T), obs::VecA(T), 
                     obs_cov::CovM(T), kwargs::StepKwargs;
                     conditioning::ConM(T)=1000.0I, 
                     m_err::ArView(T)=(1.0 ./ zeros(1,1)),
                     tol::Float64 = 0.0001,
                     j_max::Int64=40,
                     Q::CovM(T)=1.0I) where T <: Float64 

Computes the ensemble estimated gradient and Hessian terms for nonlinear least-squares
```
return ∇_J, Hess_J
```
`m_err`, `tol`, `j_max`, `Q` are optional arguments depending on the `analysis`, with 
default values provided.

Serves as an auxilliary function for IEnKS(-N), where "analysis" is a string which
determines the method of transform update ensemble Gauss-Newton calculation.  The observation
error covariance `obs_cov` is of type [`CovM`](@ref), the conditioning matrix `conditioning`
is of type [`ConM`](@ref), the keyword arguments dictionary `kwargs` is of type
[`StepKwargs`](@ref) and the model error covariance matrix `Q` is of type [`CovM`](@ref).

Currently validated `analysis` options:
 * `analysis == "ienks-bundle" || "ienks-n-bundle" || "ienks-transform" || "ienks-n-transform"`
   computes the weighted observed anomalies as per the  
   bundle or transform version of the IEnKS, described in [Bocquet &
   Sakov 2013](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.2236),
   [Grudzien, et al. 2021](https://gmd.copernicus.org/preprints/gmd-2021-306/).
   Bundle versus tranform versions of the scheme are specified by the trailing
   `analysis` string as `-bundle` or `-transform`.  The bundle version uses a small uniform 
   scalar `ϵ`, whereas the transform version uses a matrix square root inverse as the
   conditioning operator. This form of analysis differs from other schemes by returning a
   sequential-in-time value for the cost function gradient and Hessian, which will is
   utilized within the iterative smoother optimization.  A finite-size inflation scheme,
   based on the EnKF-N above, can be utilized by appending additionally a `-n` to the
   `-bundle` or `-transform` version of the IEnKS scheme specified in `analysis`.
"""
function ens_gauss_newton(analysis::String, ens::ArView(T), obs::VecA(T), 
                          obs_cov::CovM(T), kwargs::StepKwargs;
                          conditioning::ConM(T)=1000.0I, 
                          m_err::ArView(T)=(1.0 ./ zeros(1,1)),
                          tol::Float64 = 0.0001,
                          j_max::Int64=40,
                          Q::CovM(T)=1.0I) where T <: Float64 
    if analysis[1:5]=="ienks" 
        # step 0: infer observation dimension
        obs_dim = length(obs)
        
        # step 1: compute the observed ensemble and ensemble mean 
        Y = alternating_obs_operator(ens, obs_dim, kwargs)
        y_mean = mean(Y, dims=2)
        
        # step 2: compute the observed anomalies, proportional to the conditioning matrix
        # here conditioning should be supplied as trans^(-1)
        S = (Y .- y_mean) * conditioning

        # step 3: compute the cost function gradient term
        inv_obs_cov = inv(obs_cov)
        ∇J = transpose(S) * inv_obs_cov * (obs - y_mean)

        # step 4: compute the cost function gradient term
        hess_J = transpose(S) * inv_obs_cov * S

        # return tuple of the gradient and hessian terms
        return ∇J, hess_J
    end
end

##############################################################################################
""" 
    ens_update_RT!(ens::ArView(T), transform::TransM(T)) where T <: Float64

Updates forecast ensemble to the analysis ensemble by right transform (RT) method. 
```
return ens
```

Arguments include the ensemble of type [`ArView`](@ref) and the 3-tuple including the
right transform for the anomalies, the weights for the mean and the random, mean-preserving
orthogonal matrix, type [`TransM`](@ref).
"""
function ens_update_RT!(ens::ArView(T), update::TransM(T)) where T <: Float64 
    # step 0: infer dimensions and unpack the transform
    sys_dim, N_ens = size(ens)
    trans, w, U = update
    
    # step 1: compute the ensemble mean
    x_mean = mean(ens, dims=2)

    # step 2: compute the non-normalized anomalies
    X = ens .- x_mean

    # step 3: compute the update
    ens_transform = w .+ trans * U * sqrt(N_ens - 1.0)
    ens .= x_mean .+ X * ens_transform
    return ens
end


##############################################################################################
"""
    ensemble_filter(analysis::String, ens::ArView(T), obs::VecA(T), obs_cov::CovM(T),
                    s_infl::Float64, kwargs::StepKwargs) where T <: Float64 

General filter analysis step, wrapping the right transform / update, and inflation steps.
Optional keyword argument includes state_dim for extended state including parameters.
In this case, a value for the parameter covariance inflation should be included
in addition to the state covariance inflation.
```
return Dict{String,Array{Float64,2}}("ens" => ens)
```
"""
function ensemble_filter(analysis::String, ens::ArView(T), obs::VecA(T), obs_cov::CovM(T),
                         s_infl::Float64, kwargs::StepKwargs) where T <: Float64 

    # step 0: infer the system, observation and ensemble dimensions 
    sys_dim, N_ens = size(ens)
    obs_dim = length(obs)

    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]
        p_infl = kwargs["p_infl"]

    else
        state_dim = sys_dim
    end

    # step 1: compute the tranform and update ensemble
    ens_update_RT!(ens, transform_R(analysis, ens, obs, obs_cov, kwargs)) 

    # step 2a: compute multiplicative inflation of state variables
    inflate_state!(ens, s_infl, sys_dim, state_dim)

    # step 2b: if including an extended state of parameter values,
    # compute multiplicative inflation of parameter values
    if state_dim != sys_dim
        inflate_param!(ens, p_infl, sys_dim, state_dim)
    end

    return Dict{String,Array{Float64,2}}("ens" => ens)
end


##############################################################################################
"""
    ls_smoother_classic(analysis::String, ens::ArView(T), obs::ArView(T),
                        obs_cov::CovM(T), s_infl::Float64,
                        kwargs::StepKwargs) where T <: Float64 

Lag-shift ensemble Kalman smoother analysis step, classical version.

Classic EnKS uses the last filtered state for the forecast, different from the 
iterative schemes which use the once or multiple-times re-analized posterior for
the initial condition for the forecast of the states to the next shift.

Optional argument includes state dimension for extended state including parameters.
In this case, a value for the parameter covariance inflation should be included
in addition to the state covariance inflation.
```
return Dict{String,Array{Float64}}(
                                   "ens" => ens, 
                                   "post" =>  posterior, 
                                   "fore" => forecast, 
                                   "filt" => filtered
                                  ) 
```
"""    
function ls_smoother_classic(analysis::String, ens::ArView(T), obs::ArView(T),
                             obs_cov::CovM(T), s_infl::Float64,
                             kwargs::StepKwargs) where T <: Float64 
    # step 0: unpack kwargs
    f_steps = kwargs["f_steps"]::Int64
    step_model! = kwargs["step_model"]::Function
    posterior = kwargs["posterior"]::Array{Float64,3}
    
    
    # infer the ensemble, obs, and system dimensions,
    # observation sequence includes shift forward times,
    # posterior is size lag + shift
    obs_dim, shift = size(obs)
    sys_dim, N_ens, lag = size(posterior)
    lag = lag - shift

    if shift < lag
        # posterior contains length lag + shift past states, we discard the oldest shift
        # states and load the new filtered states in the routine
        posterior = cat(posterior[:, :, 1 + shift: end], 
                        Array{Float64}(undef, sys_dim, N_ens, shift), dims=3)
    end

    # optional parameter estimation
    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]::Int64
        p_infl = kwargs["p_infl"]::Float64
        p_wlk = kwargs["p_wlk"]::Float64
        param_est = true
    else
        state_dim = sys_dim
        param_est = false
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
            if param_est
                if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                    # define the diffusion structure matrix with respect to the sample value
                    # of the inertia, as per each ensemble member
                    diff_mat = zeros(20,20)
                    diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] =
                    kwargs["dx_params"]["ω"][1] ./ (2.0 * ens[21:30, j])
                    kwargs["diff_mat"] = diff_mat
                end
            end
            @views for k in 1:f_steps
                step_model!(ens[:, j], 0.0, kwargs)
                if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                    # set phase angles mod 2pi
                    ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                end
            end
        end

        # step 2b: store forecast to compute ensemble statistics before observations
        # become available
        forecast[:, :, s] = ens

        # step 2c: perform the filtering step
        trans = transform_R(analysis, ens, obs[:, s], obs_cov, kwargs)
        ens_update_RT!(ens, trans)

        # compute multiplicative inflation of state variables
        inflate_state!(ens, s_infl, sys_dim, state_dim)

        # if including an extended state of parameter values,
        # compute multiplicative inflation of parameter values
        if state_dim != sys_dim
            inflate_param!(ens, p_infl, sys_dim, state_dim)
        end

        # store the filtered states and posterior states
        filtered[:, :, s] = ens
        posterior[:, :, end - shift + s] = ens
        
        # step 2e: re-analyze the posterior in the lag window of states,
        # not including current time
        @views for l in 1:lag + s - 1 
            ens_update_RT!(posterior[:, :, l], trans)
        end
    end
            
    # step 3: if performing parameter estimation, apply the parameter model
    if state_dim != sys_dim
        param_ens = ens[state_dim + 1:end , :]
        param_mean = mean(param_ens, dims=2)
        param_ens .= param_ens + 
                     p_wlk * param_mean .* rand(Normal(), length(param_mean), N_ens)
        ens[state_dim + 1:end, :] = param_ens
    end
    
    Dict{String,Array{Float64}}(
                                "ens" => ens, 
                                "post" =>  posterior, 
                                "fore" => forecast, 
                                "filt" => filtered
                               ) 
end


##############################################################################################
"""
    ls_smoother_single_iteration(analysis::String, ens::ArView(T),
                                 obs::ArView(T), obs_cov::CovM(T),
                                 s_infl::Float64, kwargs::StepKwargs) where T <: Float64 

Lag-shift, single-iteration ensemble Kalman smoother (SIEnKS) analysis step.

Single-iteration EnKS uses the final re-analyzed posterior initial state for the forecast,
which is pushed forward in time to shift-number of observation times.
Optional argument includes state dimension for an extended state including parameters.
In this case, a value for the parameter covariance inflation should be included in
addition to the state covariance inflation.
```
return Dict{String,Array{Float64}}(
                                   "ens" => ens, 
                                   "post" =>  posterior, 
                                   "fore" => forecast, 
                                   "filt" => filtered
                                  ) 
```
"""
function ls_smoother_single_iteration(analysis::String, ens::ArView(T),
                                      obs::ArView(T), obs_cov::CovM(T),
                                      s_infl::Float64, kwargs::StepKwargs) where T <: Float64 
    # step 0: unpack kwargs, posterior contains length lag past states ending
    # with ens as final entry
    f_steps = kwargs["f_steps"]::Int64
    step_model! = kwargs["step_model"]::Function
    posterior = kwargs["posterior"]::Array{Float64,3}
    
    # infer the ensemble, obs, and system dimensions, observation sequence
    # includes lag forward times
    obs_dim, lag = size(obs)
    sys_dim, N_ens, shift = size(posterior)

    # optional parameter estimation
    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]::Int64
        p_infl = kwargs["p_infl"]::Float64
        p_wlk = kwargs["p_wlk"]::Float64
        param_est = true
    else
        state_dim = sys_dim
        param_est = false
    end

    # make a copy of the intial ens for re-analysis
    ens_0 = copy(ens)
    
    # spin to be used on the first lag-assimilations -- this makes the smoothed time-zero
    # re-analized prior the first initial condition for the future iterations
    # regardless of sda or mda settings
    spin = kwargs["spin"]::Bool
    
    # step 1: create storage for the posterior, forecast and filter values over the DAW
    # only the shift-last and shift-first values are stored as these represent the
    # newly forecasted values and last-iterate posterior estimate respectively
    if spin
        forecast = Array{Float64}(undef, sys_dim, N_ens, lag)
        filtered = Array{Float64}(undef, sys_dim, N_ens, lag)
    else
        forecast = Array{Float64}(undef, sys_dim, N_ens, shift)
        filtered = Array{Float64}(undef, sys_dim, N_ens, shift)
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
        
        # make a single iteration with SDA,
        # with MDA make a rebalancing step on the zeroth iteration
        while i <=1 
            # step 2: forward propagate the ensemble and analyze the observations
            for l in 1:lag
                # step 2a: propagate between observation times
                for j in 1:N_ens
                    if param_est
                        if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                            # define the structure matrix with respect to the sample value
                            # of the inertia, as per each ensemble member
                            diff_mat = zeros(20,20)
                            diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] =
                            kwargs["dx_params"]["ω"][1] ./ (2.0 * ens[21:30, j])
                            kwargs["diff_mat"] = diff_mat
                        end
                    end
                    @views for k in 1:f_steps
                        step_model!(ens[:, j], 0.0, kwargs)
                        if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                            # set phase angles mod 2pi
                            ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                        end
                    end
                end
                if i == 0
                    # step 2b: store forecast to compute ensemble statistics before
                    # observations become available
                    # for MDA, this is on the zeroth iteration through the DAW
                    if spin
                        # store all new forecast states
                        forecast[:, :, l] = ens
                    elseif (l > (lag - shift))
                        # only store forecasted states for beyond unobserved
                        # times beyond previous forecast windows
                        forecast[:, :, l - (lag - shift)] = ens
                    end
                    
                    # step 2c: perform the filtering step with rebalancing weights 
                    trans = transform_R(analysis,
                                      ens, obs[:, l], obs_cov * reb_weights[l], kwargs)
                    ens_update_RT!(ens, trans)

                    if spin 
                        # compute multiplicative inflation of state variables
                        inflate_state!(ens, s_infl, sys_dim, state_dim)

                        # if including an extended state of parameter values,
                        # compute multiplicative inflation of parameter values
                        if state_dim != sys_dim
                            inflate_param!(ens, p_infl, sys_dim, state_dim)
                        end
                        
                        # store all new filtered states
                        filtered[:, :, l] = ens
                    
                    elseif l > (lag - shift)
                        # store the filtered states for previously unobserved times,
                        # not mda values
                        filtered[:, :, l - (lag - shift)] = ens
                    end
                    
                    # step 2d: compute re-analyzed posterior statistics within rebalancing
                    # step, using the MDA rebalancing analysis transform for all available
                    # times on all states that will be discarded on the next shift
                    reanalysis_index = min(shift, l)
                    @views for s in 1:reanalysis_index
                        ens_update_RT!(posterior[:, :, s], trans)
                    end
                    
                    # store most recent filtered state in the posterior statistics, for all
                    # states to be discarded on the next shift > 1
                    if l < shift
                        posterior[:, :, l + 1] = ens
                    end
                else
                    # step 2c: perform the filtering step with mda weights
                    trans = transform_R(analysis,
                                      ens, obs[:, l], obs_cov * obs_weights[l], kwargs)
                    ens_update_RT!(ens, trans)
                    
                    # re-analyzed initial conditions are computed in the mda step
                    ens_update_RT!(ens_0, trans)
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
                if param_est
                    if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                        # define the structure matrix with respect to the sample value
                        # of the inertia, as per each ensemble member
                        diff_mat = zeros(20,20)
                        diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] =
                        kwargs["dx_params"]["ω"][1] ./ (2.0 * ens[21:30, j])
                        kwargs["diff_mat"] = diff_mat
                    end
                end
                @views for k in 1:f_steps
                    step_model!(ens[:, j], 0.0, kwargs)
                    if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                        # set phase angles mod 2pi
                        ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                    end
                end
            end
            if spin
                # step 2b: store forecast to compute ensemble statistics before observations
                # become available
                # if spin, store all new forecast states
                forecast[:, :, l] = ens
                
                # step 2c: apply the transformation and update step
                trans = transform_R(analysis, ens, obs[:, l], obs_cov, kwargs)
                ens_update_RT!(ens, trans)
                
                # compute multiplicative inflation of state variables
                inflate_state!(ens, s_infl, sys_dim, state_dim)

                # if including an extended state of parameter values,
                # compute multiplicative inflation of parameter values
                if state_dim != sys_dim
                    inflate_param!(ens, p_infl, sys_dim, state_dim)
                end
                
                # store all new filtered states
                filtered[:, :, l] = ens
            
                # step 2d: compute the re-analyzed initial condition if assimilation update
                ens_update_RT!(ens_0, trans)
            
            elseif l > (lag - shift)
                # step 2b: store forecast to compute ensemble statistics before observations
                # become available
                # if not spin, only store forecasted states for beyond unobserved times
                # beyond previous forecast windows
                forecast[:, :, l - (lag - shift)] = ens
                
                # step 2c: apply the transformation and update step
                trans = transform_R(analysis, ens, obs[:, l], obs_cov, kwargs)
                ens_update_RT!(ens, trans)
                
                # store the filtered states for previously unobserved times, not mda values
                filtered[:, :, l - (lag - shift)] = ens
                
                # step 2d: compute re-analyzed initial condition if assimilation update
                ens_update_RT!(ens_0, trans)
            end
        end
        # reset the ensemble with the re-analyzed prior 
        ens = copy(ens_0)
    end

    # step 3: propagate the posterior initial condition forward to the shift-forward time
    # step 3a: inflate the posterior covariance
    inflate_state!(ens, s_infl, sys_dim, state_dim)
    
    # if including an extended state of parameter values,
    # compute multiplicative inflation of parameter values
    if state_dim != sys_dim
        inflate_param!(ens, p_infl, sys_dim, state_dim)
    end

    # step 3b: if performing parameter estimation, apply the parameter model
    if state_dim != sys_dim
        param_ens = ens[state_dim + 1:end , :]
        param_mean = mean(param_ens, dims=2)
        param_ens .= param_ens +
                     p_wlk * param_mean .* rand(Normal(), length(param_mean), N_ens)
        ens[state_dim + 1:end , :] = param_ens
    end

    # step 3c: propagate the re-analyzed, resampled-in-parameter-space ensemble up by shift
    # observation times
    for s in 1:shift
        if !mda
            posterior[:, :, s] = ens
        end
        for j in 1:N_ens
            if param_est
                if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                    # define the diffusion structure matrix with respect to the sample value
                    # of the inertia, as per each ensemble member
                    diff_mat = zeros(20,20)
                    diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] =
                    kwargs["dx_params"]["ω"][1] ./ (2.0 * ens[21:30, j])
                    kwargs["diff_mat"] = diff_mat
                end
            end
            @views for k in 1:f_steps
                step_model!(ens[:, j], 0.0, kwargs)
                if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                    # set phase angles mod 2pi
                    ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                end
            end
        end
    end

    Dict{String,Array{Float64}}(
                                "ens" => ens, 
                                "post" =>  posterior, 
                                "fore" => forecast, 
                                "filt" => filtered,
                                ) 
end


##############################################################################################
"""
    ls_smoother_gauss_newton(analysis::String, ens::ArView(T),
                             obs::ArView(T), obs_cov::CovM(T), s_infl::Float64,
                             kwargs::StepKwargs; ϵ::Float64=0.0001,
                             tol::Float64=0.001, max_iter::Int64=5) where T <: Float64 

This implements a lag-shift Gauss-Newton IEnKS analysis step as in algorithm 4 of
[Bocquet & Sakov 2014](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2236).
The IEnKS uses the final re-analyzed initial state in the data assimilation window to generate
the forecast, which is subsequently pushed forward in time from the initial conidtion to
shift-number of observation times. Optional argument includes state dimension for an extended
state including parameters. In this case, a value for the parameter covariance inflation
should be included in addition to the state covariance inflation.
```
return Dict{String,Array{Float64}}(
                                   "ens" => ens, 
                                   "post" =>  posterior, 
                                   "fore" => forecast, 
                                   "filt" => filtered
                                  ) 
```
"""
function ls_smoother_gauss_newton(analysis::String, ens::ArView(T),
                                  obs::ArView(T), obs_cov::CovM(T), s_infl::Float64,
                                  kwargs::StepKwargs; ϵ::Float64=0.0001,
                                  tol::Float64=0.001, max_iter::Int64=5) where T <: Float64 
    # step 0: unpack kwargs, posterior contains length lag past states ending
    # with ens as final entry
    f_steps = kwargs["f_steps"]::Int64
    step_model! = kwargs["step_model"]::Function
    posterior = kwargs["posterior"]::Array{Float64,3}
    
    # infer the ensemble, obs, and system dimensions,
    # observation sequence includes lag forward times
    obs_dim, lag = size(obs)
    sys_dim, N_ens, shift = size(posterior)

    # optional parameter estimation
    if haskey(kwargs, "state_dim")
        state_dim = kwargs["state_dim"]::Int64
        p_infl = kwargs["p_infl"]::Float64
        p_wlk = kwargs["p_wlk"]::Float64
        param_est = true
    else
        state_dim = sys_dim
        param_est = false
    end

    # spin to be used on the first lag-assimilations -- this makes the smoothed time-zero
    # re-analized prior
    # the first initial condition for the future iterations regardless of sda or mda settings
    spin = kwargs["spin"]::Bool
    
    # step 1: create storage for the posterior filter values over the DAW, 
    # forecast values in the DAW+shift
    if spin
        forecast = Array{Float64}(undef, sys_dim, N_ens, lag + shift)
        filtered = Array{Float64}(undef, sys_dim, N_ens, lag)
    else
        forecast = Array{Float64}(undef, sys_dim, N_ens, shift)
        filtered = Array{Float64}(undef, sys_dim, N_ens, shift)
    end

    # step 1a: determine if using finite-size or MDA formalism in the below
    if analysis[1:7] == "ienks-n"
        # epsilon inflation factor corresponding to unknown forecast distribution mean
        ϵ_N = 1.0 + (1.0 / N_ens)

        # effective ensemble size
        N_effective = N_ens + 1.0
    end

    # multiple data assimilation (mda) is optional, read as boolean variable
    mda = kwargs["mda"]::Bool
    
    # algorithm splits on the use of MDA or not
    if mda
        # 1b: define the initial parameters for the two stage iterative optimization

        # define the rebalancing weights for the first sweep of the algorithm
        reb_weights = kwargs["reb_weights"]::Vector{Float64}

        # define the mda weights for the second pass of the algorithm
        obs_weights = kwargs["obs_weights"]::Vector{Float64}

        # m gives the total number of iterations of the algorithm over both the
        # rebalancing and the MDA steps, this is combined from the iteration count
        # i in each stage; the iterator i will give the number of iterations of the 
        # optimization and does not take into account the forecast / filtered iteration; 
        # for an optmized routine of the transform version, forecast / filtered statistics 
        # can be computed within the iteration count i; for the optimized bundle 
        # version, forecast / filtered statistics need to be computed with an additional 
        # iteration due to the epsilon scaling of the ensemble
        m = 0

        # stage gives the algorithm stage, 0 is rebalancing, 1 is MDA
        stage = 0

        # step 1c: compute the initial ensemble mean and normalized anomalies, 
        # and storage for the sequentially computed iterated mean, gradient 
        # and hessian terms 
        ens_mean_0 = mean(ens, dims=2)
        anom_0 = ens .- ens_mean_0 

        ∇J = Array{Float64}(undef, N_ens, lag)
        hess_J = Array{Float64}(undef, N_ens, N_ens, lag)

        # pre-allocate these variables as global for the loop re-definitions
        hessian = Symmetric(Array{Float64}(undef, N_ens, N_ens))
        new_ens = Array{Float64}(undef, sys_dim, N_ens)
        
        # step through two stages starting at zero
        while stage <=1
            # step 1d: (re)-define the conditioning for bundle versus transform varaints
            if analysis[end-5:end] == "bundle"
                trans = ϵ*I
                trans_inv = (1.0 / ϵ)*I
            elseif analysis[end-8:end] == "transform"
                trans = 1.0*I
                trans_inv = 1.0*I
            end

            # step 1e: (re)define the iteration count and the base-point for the optimization
            i = 0
            ens_mean_iter = copy(ens_mean_0) 
            w = zeros(N_ens)
            
            # step 2: begin iterative optimization
            while i < max_iter 
                # step 2a: redefine the conditioned ensemble with updated mean, after 
                # first spin run in stage 0 
                if !spin || i > 0 || stage > 0
                    ens = ens_mean_iter .+ anom_0 * trans
                end

                # step 2b: forward propagate the ensemble and sequentially store the 
                # forecast or construct cost function
                for l in 1:lag
                    # propagate between observation times
                    for j in 1:N_ens
                        if param_est
                            if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                                # define structure matrix with respect to the sample value
                                # of the inertia, as per each ensemble member
                                diff_mat = zeros(20,20)
                                diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] =
                                kwargs["dx_params"]["ω"][1] ./ (2.0 * ens[21:30, j])
                                kwargs["diff_mat"] = diff_mat
                            end
                        end
                        @views for k in 1:f_steps
                            step_model!(ens[:, j], 0.0, kwargs)
                            if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                                # set phase angles mod 2pi
                                ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                            end
                        end
                    end

                    if spin && i == 0 && stage==0
                        # if first spin, store the forecast over the entire DAW
                        forecast[:, :, l] = ens

                    # otherwise, compute the sequential terms of the gradient and hessian of 
                    # the cost function, weights depend on the stage of the algorithm
                    elseif stage == 0 
                        # this is the rebalancing step to produce filter and forecast stats
                        ∇J[:,l], hess_J[:, :, l] = ens_gauss_newton(
                                                             analysis,
                                                             ens, obs[:, l],
                                                             obs_cov * reb_weights[l], 
                                                             kwargs,
                                                             conditioning=trans_inv
                                                            )

                    elseif stage == 1
                        # this is the MDA step to shift the window forward
                        ∇J[:,l], hess_J[:, :, l] = ens_gauss_newton(
                                                             analysis,
                                                             ens,
                                                             obs[:, l],
                                                             obs_cov * obs_weights[l], 
                                                             kwargs,
                                                             conditioning=trans_inv
                                                            )
                    end

                end

                # skip this section in the first spin cycle, return and begin optimization
                if !spin || i > 0 || stage > 0
                    # step 2c: formally compute the gradient and the hessian from the 
                    # sequential components, perform Gauss-Newton after forecast iteration
                    if analysis[1:7] == "ienks-n" 
                        # use the finite size EnKF cost function for the gradient calculation 
                        ζ = 1.0 / (sum(w.^2.0) + ϵ_N)
                        gradient = N_effective * ζ * w - sum(∇J, dims=2)

                        # hessian is computed with the effective ensemble size
                        hessian = Symmetric((N_effective - 1.0) * I +
                                            dropdims(sum(hess_J, dims=3), dims=3))
                    else
                        # compute the usual cost function directly
                        gradient = (N_ens - 1.0) * w - sum(∇J, dims=2)

                        # hessian is computed with the ensemble rank
                        hessian = Symmetric((N_ens - 1.0) * I +
                                            dropdims(sum(hess_J, dims=3), dims=3))
                    end

                    if analysis[end-8:end] == "transform"
                        # transform method requires each of the below, and we make 
                        # all calculations simultaneously via the SVD for stability
                        trans, trans_inv, hessian_inv = square_root_inv(hessian, full=true)
                        
                        # compute the weights update
                        Δw = hessian_inv * gradient
                    else
                        # compute the weights update by the standard linear equation solver
                        Δw = hessian \ gradient
                    end

                    # update the weights
                    w -= Δw 

                    # update the mean via the increment, always with the zeroth 
                    # iterate of the ensemble
                    ens_mean_iter = ens_mean_0 + anom_0 * w
                    
                    if norm(Δw) < tol
                        i+=1
                        break
                    end
                end
                
                # update the iteration count
                i+=1
            end

            # step 3: compute posterior initial condiiton and propagate forward in time
            # step 3a: perform the analysis of the ensemble
            if analysis[1:7] == "ienks-n" 
                # use finite size EnKF cost function to produce adaptive
                # inflation with the hessian
                ζ = 1.0 / (sum(w.^2.0) + ϵ_N)
                hessian = Symmetric(
                                    N_effective * (ζ * I - 2.0 * ζ^(2.0) * w * transpose(w)) +
                                    dropdims(sum(hess_J, dims=3), dims=3)
                                   )
                trans = square_root_inv(hessian)
            elseif analysis == "ienks-bundle"
                trans = square_root_inv(hessian)
            end
            # compute analyzed ensemble by the iterated mean and the transformed
            # original anomalies
            U = rand_orth(N_ens)
            ens = ens_mean_iter .+ sqrt(N_ens - 1.0) * anom_0 * trans * U

            # step 3b: if performing parameter estimation, apply the parameter model
            # for the for the MDA step and shifted window
            if state_dim != sys_dim && stage == 1
                param_ens = ens[state_dim + 1:end , :]
                param_mean = mean(param_ens, dims=2)
                param_ens .= param_ens +
                             p_wlk *
                             param_mean .* rand(Normal(), length(param_mean), N_ens)
                ens[state_dim + 1:end, :] = param_ens
            end

            # step 3c: propagate the re-analyzed, resampled-in-parameter-space ensemble up 
            # by shift observation times in stage 1, store the filtered state as the forward 
            # propagated value at the new observation times within the DAW in stage 0, 
            # the forecast states as those beyond the DAW in stage 0, and
            # store the posterior at the times discarded at the next shift in stage 0
            if stage == 0
                for l in 1:lag + shift
                    if l <= shift
                        # store the posterior ensemble at times that will be discarded
                        posterior[:, :, l] = ens
                    end

                    # shift the ensemble forward Δt
                    for j in 1:N_ens
                        if param_est
                            if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                                # define structure matrix with respect to the sample value
                                # of the inertia, as per each ensemble member
                                diff_mat = zeros(20,20)
                                diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] =
                                kwargs["dx_params"]["ω"][1] ./ (2.0 * ens[21:30, j])
                                kwargs["diff_mat"] = diff_mat
                            end
                        end
                        @views for k in 1:f_steps
                            step_model!(ens[:, j], 0.0, kwargs)
                            if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                                # set phase angles mod 2pi
                                ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                            end
                        end
                    end

                    if spin && l <= lag
                        # store spin filtered states at all times up to lag
                        filtered[:, :, l] = ens
                    elseif spin && l > lag
                        # store the remaining spin forecast states at shift times
                        # beyond the DAW
                        forecast[:, :, l] = ens
                    elseif l > lag - shift && l <= lag
                        # store filtered states for newly assimilated observations
                        filtered[:, :, l - (lag - shift)] = ens
                    elseif l > lag
                        # store forecast states at shift times beyond the DAW
                        forecast[:, :, l - lag] = ens
                    end
                end
            else
                for l in 1:shift
                    for j in 1:N_ens
                        @views for k in 1:f_steps
                            step_model!(ens[:, j], 0.0, kwargs)
                            if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                                # set phase angles mod 2pi
                                ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                            end
                        end
                    end
                end
            end
            stage += 1
            m += i
        end
        
        # store and inflate the forward posterior at the new initial condition
        inflate_state!(ens, s_infl, sys_dim, state_dim)

        # if including an extended state of parameter values,
        # compute multiplicative inflation of parameter values
        if state_dim != sys_dim
            inflate_param!(ens, p_infl, sys_dim, state_dim)
        end

        Dict{String,Array{Float64}}(
                                    "ens" => ens, 
                                    "post" =>  posterior, 
                                    "fore" => forecast, 
                                    "filt" => filtered,
                                    "iterations" => Array{Float64}([m])
                                   ) 
    else
        # step 1b: define the initial correction and iteration count, note that i will
        # give the number of iterations of the optimization and does not take into
        # account the forecast / filtered iteration; for an optmized routine of the
        # transform version, forecast / filtered statistics can be computed within
        # the iteration count i; for the optimized bundle version, forecast / filtered
        # statistics need to be computed with an additional iteration due to the epsilon
        # scaling of the ensemble
        w = zeros(N_ens)
        i = 0

        # step 1c: compute the initial ensemble mean and normalized anomalies, 
        # and storage for the  sequentially computed iterated mean, gradient 
        # and hessian terms 
        ens_mean_0 = mean(ens, dims=2)
        ens_mean_iter = copy(ens_mean_0) 
        anom_0 = ens .- ens_mean_0 

        if spin 
            ∇J = Array{Float64}(undef, N_ens, lag)
            hess_J = Array{Float64}(undef, N_ens, N_ens, lag)
        else
            ∇J = Array{Float64}(undef, N_ens, shift)
            hess_J = Array{Float64}(undef, N_ens, N_ens, shift)
        end

        # pre-allocate these variables as global for the loop re-definitions
        hessian = Symmetric(Array{Float64}(undef, N_ens, N_ens))
        new_ens = Array{Float64}(undef, sys_dim, N_ens)

        # step 1e: define the conditioning for bundle versus transform varaints
        if analysis[end-5:end] == "bundle"
            trans = ϵ*I
            trans_inv = (1.0 / ϵ)*I
        elseif analysis[end-8:end] == "transform"
            trans = 1.0*I
            trans_inv = 1.0*I
        end

        # step 2: begin iterative optimization
        while i < max_iter 
            # step 2a: redefine the conditioned ensemble with updated mean, after the 
            # first spin run or for all runs if after the spin cycle
            if !spin || i > 0 
                ens = ens_mean_iter .+ anom_0 * trans
            end

            # step 2b: forward propagate the ensemble and sequentially store the forecast 
            # or construct cost function
            for l in 1:lag
                # propagate between observation times
                for j in 1:N_ens
                    if param_est
                        if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                            # define structure matrix with respect to the sample value
                            # of the inertia, as per each ensemble member
                            diff_mat = zeros(20,20)
                            diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] =
                            kwargs["dx_params"]["ω"][1] ./ (2.0 * ens[21:30, j])
                            kwargs["diff_mat"] = diff_mat
                        end
                    end
                    @views for k in 1:f_steps
                        step_model!(ens[:, j], 0.0, kwargs)
                        if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                            # set phase angles mod 2pi
                            ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                        end
                    end
                end
                if spin
                    if i == 0
                       # if first spin, store the forecast over the entire DAW
                       forecast[:, :, l] = ens
                    else
                        # otherwise, compute the sequential terms of the gradient
                        # and hessian of the cost function over all observations in the DAW
                        ∇J[:,l], hess_J[:, :, l] = ens_gauss_newton(
                                                             analysis,
                                                             ens,
                                                             obs[:, l],
                                                             obs_cov,
                                                             kwargs,
                                                             conditioning=trans_inv
                                                            )
                    end
                elseif l > (lag - shift)
                    # compute sequential terms of the gradient and hessian of the
                    # cost function only for the shift-length new observations in the DAW 
                    ∇J[:,l - (lag - shift)], 
                    hess_J[:, :, l - (lag - shift)] = ens_gauss_newton(
                                                                analysis,
                                                                ens,
                                                                obs[:, l],
                                                                obs_cov,
                                                                kwargs,
                                                                conditioning=trans_inv
                                                               )
                end
            end

            # skip this section in the first spin cycle, return and begin optimization
            if !spin || i > 0
                # step 2c: otherwise, formally compute the gradient and the hessian from the 
                # sequential components, perform Gauss-Newton step after forecast iteration
                if analysis[1:7] == "ienks-n" 
                    # use finite size EnKF cost function to produce the gradient calculation 
                    ζ = 1.0 / (sum(w.^2.0) + ϵ_N)
                    gradient = N_effective * ζ * w - sum(∇J, dims=2)

                    # hessian is computed with the effective ensemble size
                    hessian = Symmetric((N_effective - 1.0) * I +
                                        dropdims(sum(hess_J, dims=3), dims=3))
                else
                    # compute the usual cost function directly
                    gradient = (N_ens - 1.0) * w - sum(∇J, dims=2)

                    # hessian is computed with the ensemble rank
                    hessian = Symmetric((N_ens - 1.0) * I +
                                        dropdims(sum(hess_J, dims=3), dims=3))
                end
                if analysis[end-8:end] == "transform"
                    # transform method requires each of the below, and we make 
                    # all calculations simultaneously via the SVD for stability
                    trans, trans_inv, hessian_inv = square_root_inv(hessian, full=true)
                    
                    # compute the weights update
                    Δw = hessian_inv * gradient
                else
                    # compute the weights update by the standard linear equation solver
                    Δw = hessian \ gradient

                end

                # update the weights
                w -= Δw 

                # update the mean via the increment, always with the zeroth iterate 
                # of the ensemble
                ens_mean_iter = ens_mean_0 + anom_0 * w
                
                if norm(Δw) < tol
                    i +=1
                    break
                end
            end
            
            # update the iteration count
            i+=1
        end
        # step 3: compute posterior initial condiiton and propagate forward in time
        # step 3a: perform the analysis of the ensemble
        if analysis[1:7] == "ienks-n" 
            # use finite size EnKF cost function to produce adaptive inflation
            # with the hessian
            ζ = 1.0 / (sum(w.^2.0) + ϵ_N)
            hessian = Symmetric(
                                N_effective * (ζ * I - 2.0 * ζ^(2.0) * w * transpose(w)) + 
                                dropdims(sum(hess_J, dims=3), dims=3)
                               )
            
            # redefine the ensemble transform for the final update
            trans = square_root_inv(hessian)

        elseif analysis == "ienks-bundle"
            # redefine the ensemble transform for the final update,
            # this is already computed in-loop for the ienks-transform
            trans = square_root_inv(hessian)
        end

        # compute analyzed ensemble by the iterated mean and the transformed
        # original anomalies
        U = rand_orth(N_ens)
        ens = ens_mean_iter .+ sqrt(N_ens - 1.0) * anom_0 * trans * U

        # step 3b: if performing parameter estimation, apply the parameter model
        if state_dim != sys_dim
            param_ens = ens[state_dim + 1:end , :]
            param_ens = param_ens + p_wlk * rand(Normal(), size(param_ens))
            ens[state_dim + 1:end, :] = param_ens
        end

        # step 3c: propagate re-analyzed, resampled-in-parameter-space ensemble up by shift
        # observation times, store the filtered state as the forward propagated value at the 
        # new observation times within the DAW, forecast states as those beyond the DAW, and
        # store the posterior at the times discarded at the next shift
        for l in 1:lag + shift
            if l <= shift
                # store the posterior ensemble at times that will be discarded
                posterior[:, :, l] = ens
            end

            # shift the ensemble forward Δt
            for j in 1:N_ens
                if param_est
                    if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                        # define structure matrix with respect to the sample value
                        # of the inertia, as per each ensemble member
                        diff_mat = zeros(20,20)
                        diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] =
                        kwargs["dx_params"]["ω"][1] ./ (2.0 * ens[21:30, j])
                        kwargs["diff_mat"] = diff_mat
                    end
                end
                @views for k in 1:f_steps
                    step_model!(ens[:, j], 0.0, kwargs)
                    if string(parentmodule(kwargs["dx_dt"])) == "IEEE39bus"
                        # set phase angles mod 2pi
                        ens[1:10, j] .= rem2pi.(ens[1:10, j], RoundNearest)
                    end
                end
            end

            if l == shift
                # store the shift-forward ensemble for the initial condition in the new DAW
                new_ens = copy(ens)
            end

            if spin && l <= lag
                # store spin filtered states at all times up to lag
                filtered[:, :, l] = ens
            elseif spin && l > lag
                # store the remaining spin forecast states at shift times beyond the DAW
                forecast[:, :, l] = ens
            elseif l > lag - shift && l <= lag
                # store filtered states for newly assimilated observations
                filtered[:, :, l - (lag - shift)] = ens
            elseif l > lag
                # store forecast states at shift times beyond the DAW
                forecast[:, :, l - lag] = ens
            end
        end
        
        # store and inflate the forward posterior at the new initial condition
        ens = copy(new_ens)
        inflate_state!(ens, s_infl, sys_dim, state_dim)

        # if including an extended state of parameter values,
        # compute multiplicative inflation of parameter values
        if state_dim != sys_dim
            inflate_param!(ens, p_infl, sys_dim, state_dim)
        end

        Dict{String,Array{Float64}}(
                                    "ens" => ens, 
                                    "post" =>  posterior, 
                                    "fore" => forecast, 
                                    "filt" => filtered,
                                    "iterations" => Array{Float64}([i])
                                   ) 
    end
end


##############################################################################################
# end module

end
##############################################################################################
# Methods below are yet to be to debugged and benchmark
##############################################################################################
# single iteration, correlation-based lag_shift_smoother, adaptive inflation STILL DEBUGGING
#
#function ls_smoother_single_iteration_adaptive(analysis::String, ens::ArView, obs::ArView, 
#                             obs_cov::CovM, s_infl::Float64, kwargs::StepKwargs)
#
#    """Lag-shift ensemble kalman smoother analysis step, single iteration adaptive version
#
#    This version of the lag-shift enks uses the final re-analyzed posterior initial state for the forecast, 
#    which is pushed forward in time from the initial conidtion to shift-number of observation times.
#
#    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
#    case, a value for the parameter covariance inflation should be included in addition to the state covariance
#    inflation. If the analysis method is 'etks_adaptive', this utilizes the past analysis means to construct an 
#    innovation-based estimator for the model error covariances.  This is formed by the expectation step in the
#    expectation maximization algorithm dicussed by Tandeo et al. 2021."""
#    
#    # step 0: unpack kwargs, posterior contains length lag past states ending with ens as final entry
#    f_steps = kwargs["f_steps"]::Int64
#    step_model! = kwargs["step_model"]::Function
#    posterior = kwargs["posterior"]::Array{Float64,3}
#    
#    # infer the ensemble, obs, and system dimensions, observation sequence includes lag forward times
#    obs_dim, lag = size(obs)
#    sys_dim, N_ens, shift = size(posterior)
#
#    # for the adaptive inflation shceme
#    # load bool if spinning up tail of innovation statistics
#    tail_spin = kwargs["tail_spin"]::Bool
#
#    # pre_analysis will contain the sequence of the last cycle's analysis states 
#    # over the current DAW 
#    pre_analysis = kwargs["analysis"]::Array{Float64,3}
#
#    # analysis innovations contains the innovation statistics over the previous DAW plus a trail of
#    # length tail * lag to ensure more robust frequentist estimates
#    analysis_innovations = kwargs["analysis_innovations"]::ArView
#
#    # optional parameter estimation
#    if haskey(kwargs, "state_dim")
#        state_dim = kwargs["state_dim"]::Int64
#        p_infl = kwargs["p_infl"]::Float64
#        p_wlk = kwargs["p_wlk"]::Float64
#
#    else
#        state_dim = sys_dim
#    end
#
#    # make a copy of the intial ens for re-analysis
#    ens_0 = copy(ens)
#    
#    # spin to be used on the first lag-assimilations -- this makes the smoothed time-zero re-analized prior
#    # the first initial condition for the future iterations regardless of sda or mda settings
#    spin = kwargs["spin"]::Bool
#    
#    # step 1: create storage for the posterior, forecast and filter values over the DAW
#    # only the shift-last and shift-first values are stored as these represent the newly forecasted values and
#    # last-iterate posterior estimate respectively
#    if spin
#        forecast = Array{Float64}(undef, sys_dim, N_ens, lag)
#        filtered = Array{Float64}(undef, sys_dim, N_ens, lag)
#    else
#        forecast = Array{Float64}(undef, sys_dim, N_ens, shift)
#        filtered = Array{Float64}(undef, sys_dim, N_ens, shift)
#    end
#    
#    if spin
#        ### NOTE: WRITING THIS NOW SO THAT WE WILL HAVE AN ARBITRARY TAIL OF INNOVATION STATISTICS
#        # FROM THE PASS BACK THROUGH THE WINDOW, BUT WILL COMPUTE INNOVATIONS ONLY ON THE NEW 
#        # SHIFT-LENGTH REANALYSIS STATES BY THE SHIFTED DAW
#        # create storage for the analysis means computed at each forward step of the current DAW
#        post_analysis = Array{Float64}(undef, sys_dim, N_ens, lag)
#    else
#        # create storage for the analysis means computed at the shift forward states in the DAW 
#        post_analysis = Array{Float64}(undef, sys_dim, N_ens, shift)
#    end
#    
#    # step 2: forward propagate the ensemble and analyze the observations
#    for l in 1:lag
#        # step 2a: propagate between observation times
#        for j in 1:N_ens
#            @views for k in 1:f_steps
#                step_model!(ens[:, j], 0.0, kwargs)
#            end
#        end
#        if spin
#            # step 2b: store the forecast to compute ensemble statistics before observations become available
#            # if spin, store all new forecast states
#            forecast[:, :, l] = ens
#            
#            # step 2c: apply the transformation and update step
#            trans = transform_R(analysis, ens,  obs[:, l], obs_cov, kwargs)
#            ens_update_RT!(ens, trans)
#            
#            # compute multiplicative inflation of state variables
#            inflate_state!(ens, s_infl, sys_dim, state_dim)
#
#            # if including an extended state of parameter values,
#            # compute multiplicative inflation of parameter values
#            if state_dim != sys_dim
#                inflate_param!(ens, p_infl, sys_dim, state_dim)
#            end
#            
#            # store all new filtered states
#            filtered[:, :, l] = ens
#        
#            # store the re-analyzed ensembles for future statistics
#            post_analysis[:, :, l] = ens
#            for j in 1:l-1
#                post_analysis[:, :, j] = ens_update_RT!(post_analysis[:, :, j], trans)
#            end
#
#            # step 2d: compute the re-analyzed initial condition if we have an assimilation update
#            ens_update_RT!(ens_0, trans)
#        
#        elseif l > (lag - shift)
#            # step 2b: store the forecast to compute ensemble statistics before observations become available
#            # if not spin, only store forecasted states for beyond unobserved times beyond previous forecast windows
#            forecast[:, :, l - (lag - shift)] = ens
#            
#            # step 2c: apply the transformation and update step
#            if tail_spin
#                trans = transform_R(analysis, ens, obs[:, l], obs_cov, kwargs, 
#                                  m_err=analysis_innovations[:, 1:end-shift])
#            else
#                trans = transform_R(analysis, ens, obs[:, l], obs_cov, kwargs,
#                                  m_err=analysis_innovations)
#            end
#
#            ens = ens_update_RT!(ens, trans)
#            
#            # store the filtered states for previously unobserved times, not mda values
#            filtered[:, :, l - (lag - shift)] = ens
#            
#            # store the re-analyzed ensembles for future statistics
#            post_analysis[:, :, l] = ens
#            for j in 1:l-1
#                post_analysis[:, :, j] = ens_update_RT!(post_analysis[:, :, j], trans)
#            end
#
#            # step 2d: compute the re-analyzed initial condition if we have an assimilation update
#            ens_update_RT!(ens_0, trans)
#
#        elseif l > (lag - 2 * shift)
#            # store the re-analyzed ensembles for future statistics
#            post_analysis[:, :, l] = ens
#
#            # compute the innovation versus the last cycle's analysis state
#            analysis_innovations[:, :, end - lag + l] = pre_analysis[:, :, l + shift] - post_analysis[:, :, l]
#        end
#    end
#    # reset the ensemble with the re-analyzed prior 
#    ens = copy(ens_0)
#
#    # reset the analysis innovations for the next DAW
#    pre_analysis = copy(post_analysis)
#    
#    if !tail_spin 
#        # add the new shifted DAW innovations to the statistics and discard the oldest
#        # shift-innovations
#        analysis_innovations = hcat(analysis_innovations[:, shift + 1: end],
#                                    Array{Float64}(undef, sys_dim, shift))
#    end
#
#    # step 3: propagate the posterior initial condition forward to the shift-forward time
#    # step 3a: inflate the posterior covariance
#    inflate_state!(ens, s_infl, sys_dim, state_dim)
#    
#    # if including an extended state of parameter values,
#    # compute multiplicative inflation of parameter values
#    if state_dim != sys_dim
#        inflate_param!(ens, p_infl, sys_dim, state_dim)
#    end
#
#    # step 3b: if performing parameter estimation, apply the parameter model
#    if state_dim != sys_dim
#        param_ens = ens[state_dim + 1:end , :]
#        param_ens = param_ens + p_wlk * rand(Normal(), size(param_ens))
#        ens[state_dim + 1:end, :] = param_ens
#    end
#
#    # step 3c: propagate the re-analyzed, resampled-in-parameter-space ensemble up by shift
#    # observation times
#    for s in 1:shift
#        if !mda
#            posterior[:, :, s] = ens
#        end
#        for j in 1:N_ens
#            @views for k in 1:f_steps
#                step_model!(ens[:, j], 0.0, kwargs)
#            end
#        end
#    end
#
#    if tail_spin
#        # prepare storage for the new innovations concatenated to the oldest lag-innovations
#        analysis_innovations = hcat(analysis_innovations, 
#                                    Array{Float64}(undef, sys_dim, shift))
#    else
#        # reset the analysis innovations window to remove the oldest lag-innovations
#        analysis_innovations = hcat(analysis_innovations[:, shift  + 1: end], 
#                                    Array{Float64}(undef, sys_dim, lag))
#    end
#    
#    Dict{String,Array{Float64}}(
#                                "ens" => ens, 
#                                "post" =>  posterior, 
#                                "fore" => forecast, 
#                                "filt" => filtered,
#                                "anal" => pre_analysis,
#                                "inno" => analysis_innovations,
#                               )
#end
#
#
#########################################################################################################################
#########################################################################################################################
# Methods below taken from old python code, yet to completely convert, debug and benchmark
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
#    elseif analysis=="etks-adaptive"
#        ## NOTE: STILL DEVELOPMENT VERSION, NOT DEBUGGED
#        # needs to be revised for unweighted anomalies
#        # This computes the transform of the ETKF update as in Asch, Bocquet, Nodet
#        # but using a computation of the contribution of the model error covariance matrix Q
#        # in the square root as in Raanes et al. 2015 and the adaptive inflation from the
#        # frequentist estimator for the model error covariance
#        # step 0: infer the system, observation and ensemble dimensions 
#        sys_dim, N_ens = size(ens)
#        obs_dim = length(obs)
#
#        # step 1: compute the ensemble mean
#        x_mean = mean(ens, dims=2)
#
#        # step 2a: compute the normalized anomalies
#        A = (ens .- x_mean) / sqrt(N_ens - 1.0)
#
#        if !(m_err[1] == Inf)
#            # step 2b: compute the SVD for the two-sided projected model error covariance
#            F_ens = svd(A)
#            mean_err = mean(m_err, dims=2)
#
#            # NOTE: may want to consider separate formulations in which we treat
#            # the model error mean known versus unknown
#            # A_err = (m_err .- mean_err) / sqrt(length(mean_err) - 1.0)
#            A_err = m_err / sqrt(size(m_err, 2))
#            F_err = svd(A_err)
#            if N_ens <= sys_dim
#                Σ_pinv = Diagonal([1.0 ./ F_ens.S[1:N_ens-1]; 0.0]) 
#            else
#                Σ_pinv = Diagonal(1.0 ./ F_ens.S)
#            end
#
#            # step 2c: compute the square root covariance with model error anomaly
#            # contribution in the ensemble space dimension, note the difference in
#            # equation due to the normalized anomalies
#            G = Symmetric(I +  Σ_pinv * transpose(F_ens.U) * F_err.U *
#                          Diagonal(F_err.S.^2) * transpose(F_err.U) * 
#                          F_ens.U * Σ_pinv)
#            
#            G = F_ens.V * square_root(G) * F_ens.Vt
#
#            # step 2c: compute the model error adjusted anomalies
#            A = A * G
#        end
#
#        # step 3: compute the ensemble in observation space
#        Y = alternating_obs_operator(ens, obs_dim, kwargs)
#
#        # step 4: compute the ensemble mean in observation space
#        y_mean = mean(Y, dims=2)
#        
#        # step 5: compute the weighted anomalies in observation space
#        
#        # first we find the observation error covariance inverse
#        obs_sqrt_inv = square_root_inv(obs_cov)
#        
#        # then compute the weighted anomalies
#        S = (Y .- y_mean) / sqrt(N_ens - 1.0)
#        S = obs_sqrt_inv * S
#
#        # step 6: compute the weighted innovation
#        δ = obs_sqrt_inv * ( obs - y_mean )
#       
#        # step 7: compute the transform matrix
#        T = inv(Symmetric(1.0I + transpose(S) * S))
#        
#        # step 8: compute the analysis weights
#        w = T * transpose(S) * δ
#
#        # step 9: compute the square root of the transform
#        T = sqrt(T)
#        
#        # step 10:  generate mean preserving random orthogonal matrix as in sakov oke 08
#        U = rand_orth(N_ens)
#
#        # step 11: package the transform output tuple
#        T, w, U
#
#    elseif analysis=="etkf-hybrid" || analysis=="etks-hybrid"
#        # NOTE: STILL DEVELOPMENT VERSION, NOT DEBUGGED
#        # step 0: infer the system, observation and ensemble dimensions 
#        sys_dim, N_ens = size(ens)
#        obs_dim = length(obs)
#
#        # step 1: compute the background in observation space, and the square root hybrid
#        # covariance
#        Y = H * conditioning
#        x_mean = mean(ens, dims=2)
#        X = (ens .- x_mean)
#        Σ = inv(conditioning) * X
#
#        # step 2: compute the ensemble mean in observation space
#        Y_ens = H * ens
#        y_mean = mean(Y_ens, dims=2)
#        
#        # step 3: compute the sensitivity matrix in observation space
#        obs_sqrt_inv = square_root_inv(obs_cov)
#        Γ = obs_sqrt_inv * Y
#
#        # step 4: compute the weighted innovation
#        δ = obs_sqrt_inv * ( obs - y_mean )
#       
#        # step 5: run the Gauss-Newton optimization of the cost function
#
#        # step 5a: define the gradient of the cost function for the hybridized covariance
#        function ∇J!(w_full::Vector{Float64})
#            # define the factor to be inverted and compute with the SVD
#            w = w_full[1:end-2]
#            α_1 = w_full[end-1]
#            α_2 = w_full[end]
#            K = (N_ens - 1.0) / α_1 * I + transpose(Σ) * Σ 
#            F = svd(K)
#            K_inv = F.U * Diagonal(1.0 ./ F.S) * F.Vt
#            grad_w = transpose(Γ) * (δ - Γ * w) + w / α_2 - K_inv * w / α_2
#            grad_1 = 1 / α_2 * transpose(w) * K_inv * ( (1.0 - N_ens) / α_1^2.0 * I) *
#                     k_inv * w
#            grad_2 =  -transpose(w) * w / α_2^2.0 + transpose(w) * K_inv * w / α_2^2.0
#            [grad_w; grad_1; grad_2]
#        end
#
#        # step 5b: run the Gauss-Newton iteration
#        w = zeros(N_ens)
#        α_1 = 0.5
#        α_2 = 0.5
#        j = 0
#        w_full = [w; α_1; α_2]
#
#        while j < j_max
#            # compute the gradient and hessian approximation
#            grad_w = ∇J(w_full)
#            hess_w = grad_w * transpose(grad_w)
#
#            # perform Newton approximation, simultaneously computing
#            # the update transform T with the SVD based inverse at once
#            T, hessian_inv = square_root_inv(Symmetric(hess_w), inverse=true)
#            Δw = hessian_inv * grad_w
#            w_full -= Δw
#
#            if norm(Δw) < tol
#                break
#            else
#                j+=1
#            end
#        end
#
#        # step 6: store the ensemble weights
#
#        # step 6: generate mean preserving random orthogonal matrix as in sakov oke 08
#        U = rand_orth(N_ens)
#
#        # step 7: package the transform output tuple
#        T, w, U
#
#
#
