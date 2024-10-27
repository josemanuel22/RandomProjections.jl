module RandomProjections

export RandomProjection, apply_defining_function, get_powers, homopoly, linear_projection, poly_projection, circular_projection, random_slice, get_slice, max_rand_proj, max_rand_proj_with_nn

using Flux
using Random
using LinearAlgebra

mutable struct RandomProjection{T<:AbstractFloat}
    ftype::String
    nofprojections::Int
    degree::Int
    radius::T
    theta::Union{Nothing, Array{T, 2}}  # For max-RandomProjection, allows any AbstractFloat type

    function RandomProjection(ftype::String="linear", nofprojections::Int=10, degree::Int=2, radius::T=2.0) where T<:AbstractFloat
        return new{T}(ftype, nofprojections, degree, radius, nothing)
    end
end

function get_slice(rand_proj::RandomProjection, X::Array{T,2}, theta::Array{T,2}) where T <: AbstractFloat
    """
    Slices samples from distribution X~P_X based on ftype.
    Inputs:
        X: Nxd matrix of N data samples
        theta: parameters of g (e.g., a d vector in the linear case)
    """
    if rand_proj.ftype == "linear"
        return linear_projection(X, theta)
    elseif rand_proj.ftype == "poly"
        return poly_projection(rand_proj, X, theta)
    elseif rand_proj.ftype == "circular"
        return circular_projection(X, theta)
    else
        error("Defining function not implemented")
    end
end

function random_slice(rand_proj::RandomProjection, dim::Int)
    if rand_proj.ftype == "linear"
        theta = randn(rand_proj.nofprojections, dim)
        theta = hcat([th / norm(th) for th in eachcol(theta)]...)  # Normalize columns to unit length
    elseif rand_proj.ftype == "poly"
        dpoly = homopoly(dim, rand_proj.degree)
        theta = randn(rand_proj.nofprojections, dpoly)
        theta = hcat([th / norm(th) for th in eachcol(theta)]...)  # Normalize columns to unit length
    elseif rand_proj.ftype == "circular"
        theta = randn(rand_proj.nofprojections, dim)
        theta = hcat([rand_proj.radius * th / norm(th) for th in eachcol(theta)]...)  # Normalize columns to radius
    else
        error("Unsupported ftype")
    end
    return theta
end

function linear_projection(X::Array{T,2}, theta::Array{T,2}) where T <: AbstractFloat
    if size(theta, 1) == 1  # If theta is a single row vector
        return X * theta'  # Transpose theta to perform matrix multiplication
    else
        return X * theta'  # Transpose theta for general case
    end
end


function poly_projection(rand_proj::RandomProjection, X::Array{T,2}, theta::Array{T,2}) where T <: AbstractFloat
    """
    The polynomial defining function for generalized Radon transform
    Inputs
    X:  Nxd matrix of N data samples
    theta: Lxd vector that parameterizes L projections
    degree: degree of the polynomial
    """
    N, d = size(X)
    @assert size(theta, 2) == homopoly(d, rand_proj.degree)
    powers = get_powers(d, rand_proj.degree)
    HX = ones(T, N, length(powers))

    for k in 1:length(powers)
        for i in 1:d
            HX[:, k] .= HX[:, k] .* X[:, i] .^ powers[k][i]
        end
    end

    if size(theta, 1) == 1
        return HX * theta'
    else
        return HX * theta
    end
end

function circular_projection(X::Array{T,2}, theta::Array{T,2}) where T <: AbstractFloat
    """
    The circular defining function for generalized Radon transform.
    Inputs:
        X: Nxd matrix of N data samples
        theta: Lxd vector that parameterizes L projections
    """
    if size(theta, 1) == 1  # If theta is a single row vector
        return sqrt.(sum((X .- theta[1, :]) .^ 2, dims=2))  # Returns a column vector
    else
        return hcat([sqrt.(sum((X .- th') .^ 2, dims=2)) for th in eachrow(theta)]...)  # Stack each projection
    end
end

function get_powers(dim::Int, degree::Int)
    """
    Generates the powers of a homogeneous polynomial.

    Example:
    get_powers(2, 3) -> [(0, 3), (1, 2), (2, 1), (3, 0)]
    get_powers(3, 2) -> [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    """
    if dim == 1
        return [(degree,)]
    else
        result = []
        for value in 0:degree
            for permutation in get_powers(dim - 1, degree - value)
                push!(result, tuple(value, permutation...))  # Concatenating the current value with the permutation
            end
        end
        return result
    end
end

function homopoly(dim::Int, degree::Int)
    """
    Calculates the number of elements in a homogeneous polynomial.
    """
    return length(get_powers(dim, degree))
end

function max_rand_proj_with_nn(
    model,
    X::Array{T, 2},
    Y::Array{T, 2},
    loss_f,
    iterations::Int=50,
    lr::Float64=1e-4
) where T <: AbstractFloat
    """
    Maximizes loss between distributions X and Y by optimizing model parameters.
    Arguments:
        model: Neural network model to project X and Y before computing loss
        X: N x d matrix
        Y: N x d matrix
        iterations: number of gradient ascent steps
        lr: learning rate
    """
    N, dn = size(X)
    M, dm = size(Y)
    @assert dn == dm && N == M  # Ensure X and Y have compatible dimensions

    # Optimizer setup with Flux's ADAM
    optimizer = Flux.ADAM(lr)

    # Training loop
    total_loss = zeros(Float64, iterations)
    for i in 1:iterations
        # Compute loss and gradient with `withgradient`
        loss, back = Flux.withgradient(() -> begin
                X_proj = model(X)  # Project X using the neural network
                Y_proj = model(Y)  # Project Y using the neural network
                -loss_f(X_proj, Y_proj)  # Compute negative GSW loss for maximization
            end, Flux.params(model))

        total_loss[i] = -loss  # Store positive GSW for monitoring

        # Update model parameters using Flux's optimizer
        Flux.Optimise.update!(optimizer, Flux.params(model), back)
    end
    return total_loss
end

end
