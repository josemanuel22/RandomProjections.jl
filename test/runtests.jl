using RandomProjections
using Test
using LinearAlgebra
using Flux
using Statistics


# Assuming the RandomProjection struct and poly_projection function are already defined

@testset "RandomProjection struct" begin
    # Test 1: Initialization of RandomProjection struct
    rand_proj = RandomProjection("poly", 5, 3, 2.5)
    @test rand_proj.ftype == "poly"
    @test rand_proj.nofprojections == 5
    @test rand_proj.degree == 3
    @test rand_proj.radius == 2.5
    @test rand_proj.theta === nothing
end


@testset "get_powers" begin
    # Test for (dim=2, degree=3)
    @test get_powers(2, 3) == [(0, 3), (1, 2), (2, 1), (3, 0)]

    # Test for (dim=3, degree=2)
    expected_output = [
        (0, 0, 2),
        (0, 1, 1),
        (0, 2, 0),
        (1, 0, 1),
        (1, 1, 0),
        (2, 0, 0)
    ]
    @test get_powers(3, 2) == expected_output

    # Test for (dim=1, degree=2) - edge case
    @test get_powers(1, 2) == [(2,)]

    # Test for (dim=1, degree=0) - minimal edge case
    @test get_powers(1, 0) == [(0,)]

    # Test for higher degree (dim=2, degree=4)
    @test get_powers(2, 4) == [
        (0, 4),
        (1, 3),
        (2, 2),
        (3, 1),
        (4, 0)
    ]

    # Test for (dim=3, degree=1)
    @test get_powers(3, 1) == [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0)
    ]
end

@testset "homopoly Tests" begin
    # Test for (dim=2, degree=3) - should match the length of get_powers(2, 3)
    @test homopoly(2, 3) == 4

    # Test for (dim=3, degree=2) - should match the length of get_powers(3, 2)
    @test homopoly(3, 2) == 6

    # Test for (dim=1, degree=2) - edge case with a single dimension
    @test homopoly(1, 2) == 1

    # Test for (dim=1, degree=0) - minimal edge case
    @test homopoly(1, 0) == 1

    # Test for (dim=2, degree=4) - higher degree case
    @test homopoly(2, 4) == 5

    # Test for (dim=3, degree=1) - simple case with low degree
    @test homopoly(3, 1) == 3

    # Test for higher dimensions and degree (dim=3, degree=3)
    @test homopoly(3, 3) == 10

    # Test for higher dimensions and degree (dim=4, degree=2)
    @test homopoly(4, 2) == 10
end

# Test 1: linear with theta as a row vector (1D)
@testset "Single Row Vector theta" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # N=3, d=2
    theta = randn(1, size(X, 2))  # Row vector

    result = linear_projection(X, theta)

    # Check dimensions of result (N, 1)
    @test size(result) == (size(X, 1), 1)

    # Calculate expected result manually and verify
    expected_result = X * theta'
    @test isapprox(result, expected_result; atol=1e-6)
end

# Test 2: linear with theta as a matrix (2D)
@testset "Matrix theta" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # N=3, d=2
    theta = randn(3, size(X, 2))  # Matrix with multiple rows

    result = linear_projection(X, theta)

    # Check dimensions of result (N, L), where L = size(theta, 1)
    @test size(result) == (size(X, 1), size(theta, 1))

    # Calculate expected result manually and verify
    expected_result = X * theta'
    @test isapprox(result, expected_result; atol=1e-6)
end

@testset "poly_projection Tests" begin
    rand_proj = RandomProjection("poly", 5, 3, 2.5)

    # Test 2: Check poly_projection function for a known output size
    # Define a sample input matrix X and parameter theta
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # N=3, d=2
    d = size(X, 2)
    expected_dpoly = homopoly(d, rand_proj.degree)
    theta = randn(1, expected_dpoly)

    # Ensure the theta dimension matches the polynomial size
    @test size(theta, 2) == expected_dpoly

    # Execute the poly_projection function
    result = poly_projection(rand_proj, X, theta)

    # Test for correct output dimensions
    @test size(result) == (size(X, 1), size(theta, 1))

    # Test 3: Edge case with minimal input
    X_min = ones(1, 1)  # Single element matrix
    rand_proj_min = RandomProjection("poly", 1, 1, 1.0)  # Minimal RandomProjection configuration
    theta_min = randn(1, homopoly(1, rand_proj_min.degree))
    result_min = poly_projection(rand_proj_min, X_min, theta_min)

    # Ensure correct output size for minimal input
    @test size(result_min) == (1, 1)

    # Additional Test 4: Verify output type is Array{Float64, 2}
    @test typeof(result) == Array{Float64,2}
end

@testset "get_slice Function Tests" begin

    # Test 1: get_slice with 'linear' type
    @testset "Linear Type" begin
        rand_proj = RandomProjection("linear", 3, 2, 1.0)
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # N=3, d=2
        theta = randn(1, size(X, 2))

        result = get_slice(rand_proj, X, theta)

        # Check dimensions of result (N, 1)
        @test size(result) == (size(X, 1), 1)

        # Check values against direct linear computation
        expected_result = X * theta'
        @test isapprox(result, expected_result; atol=1e-6)
    end

    # Test 2: get_slice with 'poly' type
    @testset "Polynomial Type" begin
        rand_proj = RandomProjection("poly", 3, 2, 1.0)
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # N=3, d=2
        dpoly = homopoly(size(X, 2), rand_proj.degree)
        theta = randn(1, dpoly)

        result = get_slice(rand_proj, X, theta)

        # Check dimensions of result (N, 1)
        @test size(result) == (size(X, 1), 1)

        # Check values against direct poly computation
        expected_result = poly_projection(rand_proj, X, theta)
        @test isapprox(result, expected_result; atol=1e-6)
    end

    # Test 3: get_slice with 'circular' type
    @testset "Circular Type" begin
        rand_proj = RandomProjection("circular", 3, 2, 2.0)
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # N=3, d=2
        theta = randn(1, size(X, 1))

        result = get_slice(rand_proj, X, theta)

        # Check dimensions of result (N, 1)
        @test size(result) == (size(X, 1), 1)

        # Check values against direct circular computation
        expected_result = circular_projection(X, theta)
        @test isapprox(result, expected_result; atol=1e-6)
    end
end

@testset "random_slice Function Tests" begin

    # Test 1: random_slice with 'linear' type
    @testset "Linear Type" begin
        rand_proj = RandomProjection("linear", 3, 2, 1.0)
        dim = 4
        theta = random_slice(rand_proj, dim)

        # Check dimensions of theta
        @test size(theta) == (rand_proj.nofprojections, dim)

        # Check if each column is normalized to unit length
        for th in eachcol(theta)
            @test isapprox(norm(th), 1.0; atol=1e-6)
        end
    end

    # Test 2: random_slice with 'poly' type
    @testset "Polynomial Type" begin
        rand_proj = RandomProjection("poly", 3, 2, 1.0)
        dim = 3
        dpoly = homopoly(dim, rand_proj.degree)
        theta = random_slice(rand_proj, dim)

        # Check dimensions of theta
        @test size(theta) == (rand_proj.nofprojections, dpoly)

        # Check if each column is normalized to unit length
        for th in eachcol(theta)
            @test isapprox(norm(th), 1.0; atol=1e-6)
        end
    end

    # Test 3: random_slice with 'circular' type
    @testset "Circular Type" begin
        rand_proj = RandomProjection("circular", 3, 2, 2.0)
        dim = 4
        theta = random_slice(rand_proj, dim)

        # Check dimensions of theta
        @test size(theta) == (rand_proj.nofprojections, dim)

        # Check if each column is normalized to rand_proj.radius
        for th in eachcol(theta)
            @test isapprox(norm(th), rand_proj.radius; atol=1e-6)
        end
    end
end

@testset "max_rand_proj_with_nn Function Tests" begin

    # Mock loss function
    function loss_f(X_proj::Array{Float32, 2}, Y_proj::Array{Float32, 2})
        return norm(mean(X_proj - Y_proj))  # Example placeholder; replace with actual RandomProjection logic
    end

    # Define a simple neural network model for testing
    model = Chain(
        Dense(5, 10, relu),
        Dense(10, 5)
    )

    # Test 1: Basic functionality and model parameter update
    @testset "Basic Functionality" begin
        X = randn(Float32, 5, 100)  # N=100, d=5
        Y = randn(Float32, 5, 100)

        # Run max_rand_proj_with_nn and capture total loss over iterations
        total_loss = max_rand_proj_with_nn(model, X, Y, loss_f, 100, 1e-2)

        # Check that total_loss is a vector of Float64 values with correct length
        @test length(total_loss) == 100
        @test all(isa.(total_loss, Float64))

        # Check that loss generally increase over iterations
        @test mean(total_loss[1:10]) <= mean(total_loss[end-10:end])

        # Check if the final loss is a reasonable Float64 number
        @test isa(total_loss[end], Float64)
    end
end
