# Housing supply
function S_L(r,P)
    return exp.(P.alpha.*r)./(exp.(P.alpha.*r).+exp.(P.alpha.*P.p.-c));
end

# indirect utility
function U(r,a,P)
    a_U = log.(1 .+ a)*inv(Diagonal(P.sigma_s))*(P.delta_s)'
    budget = kron(ones(P.J),P.w')-kron(r,ones(P.K)')
    replace!(x -> x<=0 ? 10^-Inf -1 : x, budget)
    U = P.delta_j.+ log.(1 .+ budget) .+ a_U
    return U
end


#= indirect utility
function U(r,a,P)
    a_U = log.(1 .+ a)*inv(Diagonal(P.sigma_s))*(P.delta_s)'
    U =  P.delta_j .- log.(1 .+ r) .+ a_U
    return U
end
=#

# Demand for L-term HHs
function Static_D_L_prob(r,a,P)
    u =  U(r,a,P);
    E = exp.(u);
    norm_denominators = (ones(P.J)'*E)'
    replace!(x -> x==0 ? 1 : x, norm_denominators)
    denom = inv(Diagonal(norm_denominators));
    prob = E*denom;
    return prob
end

function Static_D_L_prob_w_outside(r,a,P)
    u =  [U(r,a,P); zeros(1,P.K)];
    E = exp.(u);
    norm_denominators = (ones(P.J+1)'*E)'
    replace!(x -> x==0 ? 1 : x, norm_denominators)
    denom = inv(Diagonal(norm_denominators));
    prob = E*denom;
    return prob
end

function Static_D_L_w_outside(r,a,P)
    D_L = Static_D_L_prob_w_outside(r,a,P)*Diagonal(P.Pop)
    return D_L
end

function Static_D_L(r,a,P)
    D_L = Static_D_L_prob(r,a,P)*Diagonal(P.Pop)
    return D_L
end

function Static_ED_vec(x,P)
    r = x[1:P.J];
    a = reshape(x[P.dim_l_a:P.dim_u_a],P.J,P.S);
    if P.outside_option
        D = Static_D_L_w_outside(r,a,P)*ones(P.K)
        D = D[1:end-1];
    else
        D = Static_D_L(r,a,P)*ones(P.K);
    end
    Static_ED_vec = 0.5*(D-S_L(r,P).*H)./(D+S_L(r,P).*H);
    return Static_ED_vec
end

function Amenity_supply(x,P)
    r = x[1:P.J];
    a = reshape(x[P.dim_l_a:P.dim_u_a],P.J,P.S);
    if P.outside_option
        D = Static_D_L_prob_w_outside(r,a,P)[1:end-1,:]*Diagonal(P.Pop)
    else
        D = Static_D_L_prob(r,a,P)*Diagonal(P.Pop)
    end
    budget = (kron(ones(P.J),P.w')-kron(ones(P.K)',r)); # assume people consume 1 unit of housing
    total_budget = lambda*D.*budget;
    exp_share = total_budget*P.delta_s;
    Amenity_supply = (exp_share ./ P.c_a_j)*inv(Diagonal(P.c_s)); # see page 31
    # c_a_j is F_{sj} and c_s is sigma_s
    return Amenity_supply
end

function Static_EA(x,P)
    Amenity_supply_vec = reshape(Amenity_supply(x,P),P.J*P.S);
    a_vec = x[P.dim_l_a:P.dim_u_a];
    Static_EA = 0.5*(Amenity_supply_vec-a_vec);
    return Static_EA
end
