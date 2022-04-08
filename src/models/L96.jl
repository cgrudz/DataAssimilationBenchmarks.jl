##############################################################################################
module L96
##############################################################################################
# imports and exports
using ..DataAssimilationBenchmarks 
export dx_dt, jacobian, l96s_tay_2_step!, ρ, α

##############################################################################################
# auxiliary function to return modular indices for the lorenz model

function mod_indx!(indx::Int64, dim::Int64)
    indx = mod(indx, dim)
    if indx==0
        indx = dim
    end
    return indx
end


##############################################################################################
# time derivative


"""
    dx_dt(x::VecA, t::Float64, dx_params::ParamDict)

    Time derivative for Lorenz-96 model, x is a single model state of size state_dim, t
    is a dummy time argument for consistency with integration methods, dx_params is a 
    parameter dictionary which is called for the forcing parameter.
"""
function dx_dt(x::VecA, t::Float64, dx_params::ParamDict)
    # unpack the (only) derivative parameter for l96
    F = dx_params["F"][1]::Float64
    x_dim = length(x)
    dx = Vector{Float64}(undef, x_dim)

    for j in 1:x_dim
        # index j minus 2, modulo the system dimension
        j_m_2 = mod_indx!(j - 2, x_dim)

        # index j minus 1, modulo the system dimension
        j_m_1 = mod_indx!(j - 1, x_dim)

        # index j plus 1, modulo the system dimension
        j_p_1 = mod_indx!(j + 1, x_dim)

        dx[j] = (x[j_p_1] - x[j_m_2])*x[j_m_1] - x[j] + F
    end
    return dx
end


##############################################################################################
# linearized time derivative

"""
    jacobian(x::Vector{Float64}, t::Float64, dx_params::ParamDict) 
    
    Computes the Jacobian of Lorenz-96 about the state x. The time variable t is a dummy
    variable for consistency with integration methods, dx_params is a parameter dictionary
    which is called for the forcing parameter. Note that this is designed to load entries in
    a zeros array and return a sparse array to make a compromise between memory
    and computational resources.
"""
function jacobian(x::Vector{Float64}, t::Float64, dx_params::ParamDict)

    x_dim = length(x)
    dxF = zeros(x_dim, x_dim)

    # looping columns j of the jacobian, loading the standard matrix
    for j in 1:x_dim

        # index j minus 2, modulo the system dimension
        j_m_2 = mod_indx!(j - 2, x_dim)

        # index j minus 1, modulo the system dimension
        j_m_1 = mod_indx!(j - 1, x_dim)

        # index j plus 1, modulo the system dimension
        j_p_1 = mod_indx!(j + 1, x_dim)

        # index j plus 2, modulo the system dimension
        j_p_2 = mod_indx!(j + 2, x_dim)

        # load the jacobian entries in corresponding rows
        dxF[j_p_2, j] = -x[j_p_1]
        dxF[j_p_1, j] = x[j_p_2] - x[j_m_1]
        dxF[j, j] = -1.0
        dxF[j_m_1, j] = x[j_m_2]
    end
    return sparse(dxF)
end

##############################################################################################
# Auxiliary functions for the 2nd order Taylor-Stratonovich expansion below. These need
# to be computed once, only as a function of the order of truncation of the Fourier
# series, p for full timeseries.

function ρ(p::Int64)
    1.0/12.0 - 0.5 * π^(-2.0) * sum(1.0 ./ Vector{Float64}(1:p).^2.0)
end

function α(p::Int64)
    (π^2.0) / 180.0 - 0.5 * π^(-2.0) * sum(1.0 ./ Vector{Float64}(1:p).^4.0)
end

##############################################################################################
# 2nd order strong taylor SDE step

"""
    l96s_tay2_step!(x::Vector{Float64}, t::Float64, kwargs::Dict{String,Any}) 
    
    One step of integration rule for l96 second order taylor rule
    The ρ and α are to be computed by the auxiliary functions in the L96 submodule, depending
    only on p, and supplied for all steps. This is the general formulation which includes,
    eg. dependence on the truncation of terms in the auxilliary function C with
    respect to the parameter p.  In general, truncation at p=1 is all that is
    necessary for order 2.0 convergence, and in this case C below is identically
    equal to zero.  This auxilliary function can be removed (and is removed) in other
    implementations for simplicity.
    
    This method is derived in
    Grudzien, C. et al.: On the numerical integration of the Lorenz-96 model,
    with scalar additive noise, for benchmark twin experiments,
    Geosci. Model Dev., 13, 1903–1924, https://doi.org/10.5194/gmd-13-1903-2020, 2020.
    NOTE: this Julia version still pending validation as in the above manuscript
"""
function l96s_tay2_step!(x::Vector{Float64}, t::Float64, kwargs::Dict{String,Any})

    # Infer model and parameters
    sys_dim = length(x)
    dx_params = kwargs["dx_params"]::ParamDict
    h = kwargs["h"]::Float64
    diffusion = kwargs["diffusion"]::Float64
    p = kwargs["p"]::Int64
    ρ = kwargs["ρ"]::Float64
    α = kwargs["α"]::Float64

    # Compute the deterministic dxdt and the jacobian equations
    dx = dx_dt(x, 0.0, dx_params)
    Jac_x = jacobian(x, 0.0, dx_params)

    ## random variables
    # Vectors ξ, μ, ϕ are sys_dim X 1 vectors of iid standard normal variables,
    # ζ and η are sys_dim X p matrices of iid standard normal variables.
    # Functional relationships describe each variable W_j as the transformation of
    # ξ_j to be of variace given by the length of the time step h. Functions of random
    # Fourier coefficients a_i, b_i are given in terms μ / η and ϕ / ζ respectively.

    # draw standard normal samples
    rndm = rand(Normal(), sys_dim, 2*p + 3)
    ξ = rndm[:, 1]

    μ = rndm[:, 2]
    ϕ = rndm[:, 3]

    ζ = rndm[:, 4: p+3]
    η = rndm[:, p+4: end]

    ### define the auxiliary functions of random fourier coefficients, a and b

    # denominators for the a series
    denoms = repeat((1.0 ./ Vector{Float64}(1:p)), 1, sys_dim)

    # vector of sums defining a terms
    a = -2.0 * sqrt(h * ρ) * μ - sqrt(2.0*h) * sum(ζ' .* denoms, dims=1)' / π

    # denominators for the b series
    denoms = repeat((1.0 ./ Vector{Float64}(1:p).^2.0), 1, sys_dim)

    # vector of sums defining b terms
    b = sqrt(h * α) * ϕ + sqrt(h / (2.0 * π^2.0) ) * sum(η' .* denoms, dims=1)'

    # vector of first order Stratonovich integrals
    J_pdelta = (h / 2.0) * (sqrt(h) * ξ + a)

    ### auxiliary functions for higher order stratonovich integrals ###
    # C function is optional for higher precision but does not change order of convergence
    function C(l, j)
        if p == 1
            return 0.0
        end
        c = zeros(p, p)
        # define the coefficient as a sum of matrix entries where r and k do not agree
        indx = Set(1:p)

        for r in 1:p
            # vals are all values not equal to r
            vals = setdiff(indx, Set(r))
            for k in vals
                # and for column r, define all row entries below, with zeros on diagonal
                c[k, r] = (r / (r^2 - k^2)) * ((1.0 / k) * ζ[l, r] * ζ[j, k] + (1.0 / r) *
                                               η[l, r] * η[j, k])
            end
        end

        # return the sum of all values scaled by -1/(2pi^2)
        -0.5 * π^(-2.0) * sum(c)
    end

    function Ψ(l, j)
        # Ψ - generic function of the indicies l and j, define Ψ plus and Ψ minus index-wise
        h^2.0 * ξ[l] * ξ[j] / 3.0 + h * a[l] * a[j] / 2.0 + 
        h^(1.5) * (ξ[l] * a[j] + ξ[j] * a[l]) / 4.0 -
        h^(1.5) * (ξ[l] * b[j] + ξ[j] * b[l]) / (2.0 * π) - h^2.0 * (C(l,j) + C(j,l))
    end

    # define the approximations of the second order Stratonovich integral
    Ψ_plus = Vector{Float64}(undef, sys_dim)
    Ψ_minus = Vector{Float64}(undef, sys_dim)
    for i in 1:sys_dim
        Ψ_plus[i] = Ψ(mod_indx!((i-1), sys_dim), mod_indx!((i+1), sys_dim))
        Ψ_minus[i] = Ψ(mod_indx!((i-2), sys_dim), mod_indx!((i-1), sys_dim))
    end

    # the final vectorized step forward is given as
    x .= collect(Iterators.flatten(
            x + dx * h + h^2.0 * 0.5 * Jac_x * dx +  # deterministic taylor step
            diffusion * sqrt(h) * ξ +                # stochastic euler step
            diffusion * Jac_x * J_pdelta +           # stochastic first order taylor step
            diffusion^2.0 * (Ψ_plus - Ψ_minus)       # stochastic second order taylor step
           ))
end

##############################################################################################
# end module

end
