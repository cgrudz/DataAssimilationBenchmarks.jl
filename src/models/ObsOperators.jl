##############################################################################################
module ObsOperators
##############################################################################################
# imports and exports
using LinearAlgebra, SparseArrays
using ..DataAssimilationBenchmarks
export alternating_projector, alternating_obs_operator, alternating_obs_operator_jacobian
##############################################################################################
# Main methods
##############################################################################################
"""
    alternating_projector(x::VecA(T), obs_dim::Int64) where T <: Real
    alternating_projector(ens::ArView(T), obs_dim::Int64) where T <: Real

Utility method produces a projection of alternating vector or ensemble components via slicing.

```
return x
return ens
```

This operator takes a single model state `x` of type [`VecA`](@ref), a truth twin time series
or an ensemble of states of type [`ArView`](@ref), and maps this data to alternating
row components.  If truth twin is 2D, then the first index corresponds to the state dimension
and the second index corresponds to the time dimension.  The ensemble is assumed to be
2D where the first index corresponds to the state dimension and the second index
corresponds to the ensemble dimension.

The operator selects row components of the input to keep based on the `obs_dim`.
States correpsonding to even state dimension indices
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
@doc raw"""
    alternating_obs_operator(x::VecA(T), obs_dim::Int64, kwargs::StepKwargs) where T <: Real
    alternating_obs_operator(ens::ArView(T), obs_dim::Int64,
                             kwargs::StepKwargs) where T <: Real

This produces observations of alternating state vector components for generating pseudo-data.
```
return obs
```

This operator takes a single model state `x` of type [`VecA`](@ref), a truth twin time series
or an ensemble of states of type [`ArView`](@ref), and maps this data to the observation
space via the method [`alternating_projector`](@ref) and (possibly) a nonlinear transform.
The truth twin in this version is assumed to be 2D, where the first index corresponds to
the state dimension and the second index corresponds to the time dimension.  The ensemble
is assumed to be 2D where the first index corresponds to the state dimension and the
second index corresponds to the ensemble dimension.
The `γ` parameter (optional) in `kwargs` of type  [`StepKwargs`](@ref) controls the
component-wise transformation of the remaining state vector components mapped to the
observation space.  For `γ=1.0`, there is no transformation applied, and the observation
operator acts as a linear projection onto the remaining components of the state vector,
equivalent to not specifying `γ`. For `γ>1.0`, the nonlinear observation operator of
[Asch, et al. (2016).](https://epubs.siam.org/doi/book/10.1137/1.9781611974546),
pg. 181 is applied,
```math
\begin{align}
\mathcal{H}(\pmb{x}) = \frac{\pmb{x}}{2}\circ\left[\pmb{1} + \left(\frac{\vert\pmb{x}\vert}{10} \right)^{\gamma - 1}\right]
\end{align}
```
where ``\circ`` is the Schur product, and which limits to the identity for `γ=1.0`.
If `γ=0.0`, the quadratic observation operator of
[Hoteit, et al. (2012).](https://journals.ametsoc.org/view/journals/mwre/140/2/2011mwr3640.1.xml),
```math
\begin{align}
\mathcal{H}(\pmb{x}) =0.05 \pmb{x} \circ \pmb{x}
\end{align}
```
is applied to the remaining state components (note, this is not a continuous limit).
If `γ<0.0`, the exponential observation
operator of [Wu, et al. (2014).](https://npg.copernicus.org/articles/21/955/2014/)
```math
\begin{align}
\mathcal{H}(\pmb{x}) = \pmb{x} \circ \exp\{- \gamma \pmb{x} \}
\end{align}
```
is applied to the remaining state vector components, where the exponential
is applied componentwise (note, this is also not a continuous limit).
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
            obs .= (obs ./ 2.0) .* ( 1.0 .+ ( abs.(obs) ./ 10.0 ).^(γ - 1.0) )

        elseif γ == 0.0
            obs .= 0.05*obs.^2.0

        elseif γ < 0.0
            obs .= obs .* exp.(-γ * obs)
        end
    end
    return obs
end

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
    alternating_obs_operator_jacobian(x::VecA(T), obs_dim::Int64,
    kwargs::StepKwargs) where T <: Real

Explicitly computes the jacobian of the alternating observation operator
given a single model state `x` of type [`VecA`](@ref) and desired dimension of observations
'obs_dim' for the jacobian. The `γ` parameter (optional) in `kwargs` of type
[`StepKwargs`](@ref) controls the component-wise transformation of the remaining state
vector components mapped to the observation space.  For `γ=1.0`, there is no
transformation applied, and the observation operator acts as a linear projection onto
the remaining components of the state vector, equivalent to not specifying `γ`.
For `γ>1.0`, the nonlinear observation operator of
[Asch, et al. (2016).](https://epubs.siam.org/doi/book/10.1137/1.9781611974546),
pg. 181 is applied, which limits to the identity for `γ=1.0`.  If `γ=0.0`, the quadratic
observation operator of [Hoteit, et al. (2012).](https://journals.ametsoc.org/view/journals/mwre/140/2/2011mwr3640.1.xml)
is applied to the remaining state components.  If `γ<0.0`, the exponential observation
operator of [Wu, et al. (2014).](https://npg.copernicus.org/articles/21/955/2014/)
is applied to the remaining state vector components.
"""

function alternating_obs_operator_jacobian(x::VecA(T), obs_dim::Int64,
    kwargs::StepKwargs) where T <: Real
    sys_dim = length(x)
    if haskey(kwargs, "state_dim")
        # performing parameter estimation, load the dynamic state dimension
        state_dim = kwargs["state_dim"]::Int64

        # observation operator for extended state, without observing extended state components
        jac = copy(x[1:state_dim])

        # proceed with alternating observations of the regular state vector
        sys_dim = state_dim
    else
        jac = copy(x)
    end

    # jacobian calculation
    if haskey(kwargs, "γ")
        γ = kwargs["γ"]::Float64
        if γ > 1.0
            jac .= (1.0 / 2.0) .* (((jac .*(γ - 1.0) / 10.0) .* (( abs.(jac) / 10.0 ).^(γ - 2.0))) .+ 1.0 .+ (( abs.(jac) / 10.0 ).^(γ - 1.0)))

        elseif γ == 0.0
            jac = 0.1.*jac

        elseif γ < 0.0
            jac .= exp.(-γ * jac) .* (1.0 .- (γ * jac))

        end
    end

    # matrix formation and projection
    jacobian_matrix = alternating_projector(diagm(jac), obs_dim)

    return jacobian_matrix
end
##############################################################################################
# end module

end
