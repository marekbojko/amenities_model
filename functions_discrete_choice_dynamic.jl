function S_L(r,P)
    return exp.(P.alpha.*r)./(exp.(P.alpha.*r).+exp.(P.alpha.*P.p.-P.c));
end

"""
function MC(P)

Computes the moving cost for the specified parameters in the named tuple P
"""
function MC(P)
    # "Moving" options have costs based on the distance between locations
    MC_inside_options =  P.m_0 .+ P.m_1*P.dist_mat
    # Include the "staying put" option - there is no moving cost
    MC = [MC_inside_options; zeros(P.J+1)']
    return MC
end

function compose_mat_for_utility(v,f::Function,n_col::Int64)
    return [mat_vals_by_row(v,f,n_col); zeros(Int64,n_col)'; [f.(v);0]']
end

function flow_utility_one_type(x,k,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)
    U = compose_mat_for_utility(log.(a)*(P.delta_a[k,:] ./ P.sigma_s),x -> x,P.J+1) +
        compose_mat_for_utility(P.delta_j,x -> x, P.J+1) - MC(P)
            # .+ P.delta_p[k]*compose_mat_for_utility(1 .+ P.p,log,P.J+1)
    if P.w_in_util
        budget = P.w[k]*ones(P.J)-r
        replace!(x -> x<=0 ? 10^-Inf : x, budget)
        U_ret = U +  P.delta_r[k]*compose_mat_for_utility(budget,log,P.J+1)
    else
        U_ret = U .+ P.delta_w[k] * compose_mat_for_utility(ones(P.J),x -> x,P.J+1) * log(P.w[k])
            .- P.delta_r[k]*compose_mat_for_utility(r,log,P.J+1)
    end
    return U_ret
end

function flow_utility_all_types(x,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)
    U = ones(P.J+2,P.J+1,P.K)
    for k in 1:P.K
        U[:,:,k] = flow_utility_one_type(x,k,P)
    end
    return U
end

function value_function_one_type(x,k,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)

    # initial guess
    EV_G = rand(P.J+1)
    dist_func = 1
    iter = 0

    # iterate
    if P.m_0 > 0
        while dist_func > P.tol && iter <= P.max_iter
            DEV = [mat_vals_by_row(EV_G, x -> x, P.J+1);EV_G']
            v = flow_utility_one_type(x,k,P) + P.beta*DEV
            EV = log.(exp.(v')*ones(P.D))
            dist_func = norm(EV-EV_G,Inf)
            EV_G = EV[:]
            iter += 1
        end
    elseif P.m_0 == 0
        while dist_func > P.tol && iter <= P.max_iter
            DEV = mat_vals_by_row(EV_G, x -> x, P.J+1)
            v = flow_utility_one_type(x,k,P)[1:end-1,:] + P.beta*DEV
            EV = log.(exp.(v')*ones(P.J+1))
            dist_func = norm(EV-EV_G,Inf)
            EV_G = EV[:]
            iter += 1
        end
    end
    return EV_G
end

function value_function_all_types(x,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)
    EV = ones(P.J+1,P.K)
    for k in 1:P.K
        EV[:,k] = value_function_one_type(x,k,P)
    end
    return EV
end

function transition_matrix_loc_action_one_type(x,k,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)
    EV_k = value_function_one_type(x,k,P)
    if P.m_0 > 0
        DEV = [mat_vals_by_row(EV_k, x -> x, P.J+1);EV_k']
        v = flow_utility_one_type(x,k,P) + P.beta*DEV
        denom = exp.(v)' * ones(P.D)
        trans_mat = exp.(v)*inv(Diagonal(denom))
        return trans_mat' #(J+1)xD matrix
    elseif P.m_0 == 0
        v = flow_utility_one_type(x,k,P)[1:end-1,1] + P.beta*EV_k
        denom = exp.(v)' * ones(P.J+1)
        prob_vec = exp.(v)/denom
        trans_mat = mat_vals_by_row(prob_vec, x -> x, P.J+1)
    else
        return
    end
    return trans_mat' #(J+1)xD matrix
end

function transition_matrix_loc_to_loc_one_type(x,k,P)
    if P.m_0 > 0
        trans_mat_d_j = transition_matrix_loc_action_one_type(x,k,P)
        trans_mat_own = trans_mat_d_j[:,1:P.J+1]
        staying_put_prob_vec = trans_mat_d_j[:,end]
        trans_mat_loc_to_loc = trans_mat_own + Diagonal(staying_put_prob_vec)
        # there could be some rounding errors, so we need to make sure the matrix
        # is row-stochastic
        row_sums = trans_mat_loc_to_loc*ones(P.J+1)
        trans_mat_loc_to_loc = trans_mat_loc_to_loc ./ row_sums
        return trans_mat_loc_to_loc
    elseif P.m_0 == 0
        return transition_matrix_loc_action_one_type(x,k,P)
    else
        return
    end
end

function trans_mat_loc_loc_one_type_noinf(x,k,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)

    ### Compute flow utility for x
    flow_U = compose_mat_for_utility(log.(a)*(P.delta_a[k,:] ./ P.sigma_s),x -> x,P.J+1) +
        compose_mat_for_utility(P.delta_j,x -> x, P.J+1) - MC(P)
            # .+ P.delta_p[k]*compose_mat_for_utility(1 .+ P.p,log,P.J+1)
    if P.w_in_util
        budget = P.w[k]*ones(P.J)-r
        z = replace(x -> x<=0 ? 1 : 0, budget)
        replace!(x -> x<=0 ? 10^(-128) : x, budget)
        flow_U = flow_U +  P.delta_r[k]*compose_mat_for_utility(budget,log,P.J+1)
    else
        flow_U = flow_U .+ P.delta_w[k] * compose_mat_for_utility(ones(P.J),x -> x,P.J+1) * log(P.w[k])
            .- P.delta_r[k]*compose_mat_for_utility(r,log,P.J+1)
    end

    # initial guess
    EV_G = [rand(P.J);0]
    dist_func = 1
    iter = 0

    if P.m_0 > 0

        ### Compute the value function
        # iterate
        while dist_func > P.tol && iter <= P.max_iter
            DEV = [mat_vals_by_row(EV_G, x -> x, P.J+1);EV_G']
            v = flow_U + P.beta*DEV
            EV = log.(exp.(v')*ones(P.D))
            dist_func = norm(EV-EV_G,Inf)
            EV_G = EV[:]
            iter += 1
        end

        ### Compute action-loc transition matrix
        DEV = [mat_vals_by_row(EV_G, x -> x, P.J+1);EV_G']
        v = flow_U + P.beta*DEV
        denom = (exp.(v) .* (1 .- compose_mat_for_utility(z, x -> x, P.J+1)))' * ones(P.D)
        trans_mat = (exp.(v) .* (1 .- compose_mat_for_utility(z, x -> x, P.J+1)))*inv(Diagonal(denom))
        trans_mat_d_j = trans_mat'
        trans_mat_own = trans_mat_d_j[:,1:P.J+1]
        staying_put_prob_vec = trans_mat_d_j[:,end]
        trans_mat_loc_to_loc = trans_mat_own + Diagonal(staying_put_prob_vec)

        # there could be some rounding errors, so we need to make sure the matrix
        # is row-stochastic
        row_sums = trans_mat_loc_to_loc*ones(P.J+1)
        trans_mat_loc_to_loc = trans_mat_loc_to_loc ./ row_sums
        return trans_mat_loc_to_loc

    elseif P.m_0 == 0

        ### Compute the value function
        # iterate
        while dist_func > P.tol && iter <= P.max_iter
            DEV = mat_vals_by_row(EV_G, x -> x, P.J+1)
            v = flow_U[1:end-1,:] + P.beta*DEV
            EV = log.(exp.(v')*ones(P.J+1))
            dist_func = norm(EV-EV_G,Inf)
            EV_G = EV[:]
            iter += 1
        end

        v = flow_U[1:P.J+1,1] + P.beta*EV_G
        z = [z; 0]
        denom = (exp.(v) .* (1 .- z))' * ones(P.J+1)
        prob_vec = (exp.(v) .* (1 .- z))/denom
        trans_mat = mat_vals_by_row(prob_vec, x -> x, P.J+1)
        return trans_mat'
    else
        return
    end
end

function stationary_dist_one_type(x,k,P)
    return stationary_dist_MC_by_iter(trans_mat_loc_loc_one_type_noinf(x,k,P))
    #return stationary_dist_MC_by_iter(transition_matrix_loc_to_loc_one_type(x,k,P))
end

function stationary_dist_all_types(x,P)
    stationary_dist_types = [stationary_dist_one_type(x,k,P) for k in 1:P.K]
    stationary_dist_types = hcat(stationary_dist_types...)
    return stationary_dist_types
end

function D_L(x,P)
    # Stack the stationary distributions for each type into a matrix
    stationary_dist_types = stationary_dist_all_types(x,P)
    # The total demand is equal to linear combination of these vectors weighted
    # by the number of agents of the given types
    D_L = stationary_dist_types*Diagonal(P.Pop)
    return D_L
end

function ED_L(x,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)
    D = D_L(x,P)[1:P.J,:]*ones(P.K)
    Static_ED_vec = D-S_L(r,P) .* P.H
    return Static_ED_vec
end

function Amenity_supply(x,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)
    D = D_L(x,P)[1:P.J,:]
    budget = (kron(ones(P.J),P.w')-kron(ones(P.K)',r)) # assume people consume 1 unit of housing
    total_budget = P.lambda * D.*budget
    exp_share = total_budget*P.delta_a
    Amenity_supply = (exp_share ./ P.c_a_j)*inv(Diagonal(P.sigma_s))
    return Amenity_supply
end

function EA_S(x,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)
    Static_EA = Amenity_supply(x,P) - a
    Static_EA = reshape(Static_EA,P.J*P.S)
    return Static_EA
end

function res_func(x,P)
    ED = ED_L(x,P)
    EA = EA_S(x,P)
    SSR_housing = ED'ED
    SSR_amenities = EA'EA
    return SSR_housing + SSR_amenities
end

function welfare_households(x,P)
    EV = value_function_all_types(x,P)
    stationary_dist_types = stationary_dist_all_types(x,P)
    return P.Pop .* ((stationary_dist_types .* EV)' * ones(P.J+1))
end

function welfare_landlords(x,P)
    r,a = unpack_vec(x,P.J,P.J,P.S)
    return log.(exp.(P.alpha*r)+exp.(P.alpha*P.p .- P.c)) .* P.H
end
