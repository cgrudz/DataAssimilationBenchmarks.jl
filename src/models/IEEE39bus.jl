########################################################################################################################
module IEEE39bus
########################################################################################################################
# imports and exports
using Debugger
using Random, Distributions
using LinearAlgebra
export dx_dt

########################################################################################################################
########################################################################################################################
# Type union declarations for multiple dispatch
# and type aliases

# vectors and ensemble members of sample
VecA = Union{Vector{Float64}, SubArray{Float64, 1}}

# dictionary for model parameters
ParamDict = Union{Dict{String, Array{Float64}}, Dict{String, Vector{Float64}}}

########################################################################################################################
########################################################################################################################

function dx_dt(x::T, t::Float64, dx_params::ParamDict) where {T <: VecA}

    """Function defining the time derivative of the phase and fequency of the effective-network swing eq model

    The input x is a 2 n_g vector of the phase and fequency at each of the n_g generator buses. The input
    dx_params is a ParamDict of all system parameters to be passed to the integration scheme.  
    The system is currenty defined autonomously to be run as an SDE, noise perturbed steady state."""

    # unpack the system parameters effective network of
    # Nishikawa, T., & Motter, A. E. (2015). Comparative analysis of existing models for power-grid synchronization.
    A = dx_params["A"]::Array{Float64}
    D = dx_params["D"]::Array{Float64}
    H = dx_params["H"]::Array{Float64}
    K = dx_params["K"]::Array{Float64}
    γ = dx_params["γ"]::Array{Float64}
    ω = dx_params["ω"]::Array{Float64}

    # we convert the effective bus coupling and passive injection to contain the change of variable terms
    K = ω[1] * K / 2.0 
    A = ω[1] * A / 2.0

    # unpack the phase and frequency at the n_g buses, with all phases listed first, then all
    # fequencies in the order of the bus index
    n_g = convert(Int, length(x) / 2)
    δ_1 = @view x[1:n_g]
    δ_2 = @view x[n_g+1:end]

    # define the vector of the derivatives
    dx = zeros(2 * n_g)

    # derivative of the phase equals frequency
    dx[1:n_g] .= δ_2

    # compute the derivative of the inertia normalized frequencies
    # entry j is defined as A_j * ω/2 - D_j /2 * δ_2 - Σ_{i!=j} K * ω/2 * sin(δ_j - δ_i - γ_ij)
    for j in 1:n_g
        for i in 1:n_g
            if j != i
                # K is symmetric, we loop over the columns for faster memory access
                # with the same variable j as in the row index of the derivative
                dx[n_g + j] += -K[i, j] * sin(δ_1[j] - δ_1[i] - γ[i, j])
            end
        end
        # finally apply the remaining terms
        dx[n_g + j] += A[j] - δ_2[j] * D[j] / 2.0
    end
    # to compute the derivative of the frequencies, we finally 
    # divide back out by the inertia
    dx[n_g + 1 : end] = dx[n_g + 1: end] ./ H
    return dx
end

########################################################################################################################

end
