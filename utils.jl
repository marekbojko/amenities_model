"""
function approx_equal(v1::Vector,v2::Vector)

Compares two vectors of equal length entry-wise given a tolerable distance.
"""

function approx_equal(v1::Vector,v2::Vector,ϵ=0.001)
    if length(v1)!=length(v2)
        return 0
    else
        return all([abs(v1[i] - v2[i]) <= ϵ for i in 1:length(v1)])
    end
end


"""
function dissimilarity_index(r::Vector,p::Vector)

Computes the dissimilarity index for two types of households. See
    https://en.wikipedia.org/wiki/Index_of_dissimilarity for details.
"""
function dissimilarity_index(r::Vector,p::Vector)
    normalize!(r,1)
    normalize!(p,1)
    return 1/2*norm(r-p,1)
end


"""
function get_ordered_indices(v::Vector, ascending = true)

Returns indices of elements of the original vector in the order in which
they would appear if the vector was orded
"""
function get_ordered_indices(v::Vector, ascending = true)
    if ascending
        sorted_v = sort(v)
        return [findfirst(x -> x==el, sorted_v) for el in v]
    elseif ascending == false
        sorted_v = sort(v, reverse = true)
        return [findfirst(x -> x==el, sorted_v) for el in v]
    end
end

"""
function generate_rand_pos_def_symmetric_mat_w_0_diag(n::Int64)

Generates a pseudo-random symmetric positive-definite matrix with 0 diagonal elements.
"""
function generate_rand_pos_def_symmetric_mat_w_0_diag(n::Int64)
    # Generate a random matrix
    X = rand(n,n)
    # Transform to a symmetric matrix
    A = X'X
    # Set diagonal elements to 0
    A[diagind(A)] .= 0
    return A
end


"""
function mat_ones_except_row(n_row::Int64, n_col::Int64, row_zeros::Int64)

Generates a matrix of ones of the specified dimensions except for a specified
    row which is a row of zeros
"""
function mat_ones_except_row(n_row::Int64, n_col::Int64, row_zeros::Int64)
    mat_above = ones(Int64,row_zeros - 1, n_col)
    mat_below = ones(Int64,n_row - row_zeros, n_col)
    return [mat_above; zeros(Int64, n_col)'; mat_below]
end


"""
function mat_vals_by_row(v::Vector,f::Function,n_col::Int64)

Given a vector and a function, returns a matrix of dimensions length(v)×n_col
    where each entry of row i is the function f applied to the i-th element of v
"""
function mat_vals_by_row(v::Vector,f::Function,n_col::Int64)
        # Create a stack of vectors
        vecs_stacked = [f.(v[i])*ones(Int64, n_col)' for i in 1:length(v)]
        # Change to a matrix
        return vcat(vecs_stacked...)
end

"""
function stationary_dist_MC(M::AbstractMatrix{Float64})

Computes the stationary distribution of a Markov Chain given its transition matrix
"""
<<<<<<< HEAD
function stationary_dist_MC(M)
=======
function stationary_dist_MC(M::AbstractMatrix{Float64})
>>>>>>> 165923626c5fc839d6fe13eb9ef5ef55720d2afe
    # Compute eigenvalues and eigenvectors
    eigen_vals, eigen_vecs = eigen(M')

    # Find the eigenvector corresponding to eigenvalue 1 and corresponding eigenvector
    unit_eigenval_ind = findfirst(x -> x ≈ 1.0,eigen_vals)
    unit_eigenvec = abs.(eigen_vecs[:,unit_eigenval_ind])

    # Normalize and return
    return unit_eigenvec / norm(unit_eigenvec,1)
end

"""
function stationary_dist_MC_by_iter(M::AbstractMatrix{Float64})

Finds the stationary distribution of a Markov Chain by rising the transition matrix to a high power
"""
function stationary_dist_MC_by_iter(M, tol::Float64 = 10^-8, n_iter::Int64 = 10^5)
    counter = 0
    dist = 1
    while dist > tol && counter <= n_iter
        M_pow = M*M
        dist = norm(M_pow - M, Inf)
        counter += 1
        M = M_pow
    end
    return (normalize(ones(size(M,1)),1)' * M)'
end

"""
function unpack_vec(x::Vector{Float64},dim_vec::Int64,n_row_mat::Int64,n_col_mat::Int64)

Unpacks a vector into a vector (first dim_vec elements) and a matrix (with n_row_mat rows
    and n_col_mat columns)
"""
function unpack_vec(x,dim_vec::Int64,n_row_mat::Int64,n_col_mat::Int64)
    vec = x[1:dim_vec]
    mat = reshape(x[dim_vec+1:end],n_row_mat,n_col_mat)
    return vec, mat
end
