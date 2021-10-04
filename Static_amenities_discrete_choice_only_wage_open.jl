cd("/home/mbojko/amenities_model")

include("functions_discrete_choice_housing.jl")
include("utils.jl")

using Random, LinearAlgebra, Optim, ForwardDiff, BenchmarkTools, DataFrames
using Optim: converged, maximum, maximizer, minimizer, iterations
using CSV, Plots
using Plots.PlotMeasures

Random.seed!(1234);

# Number of groups, number of amenities, number of locations
K = 2;
S = 2;
J = 5;

# Helper params
dim_l_a  = J+1;
dim_u_a  = J*(1+S);

# Population
Pop = 30*J*[1/2, 1/2];

# Closed or open city?
outside_option = true

# Basic Params
P = (K = K,
     S = S,
     J = J,
     Pop = Pop,
     outside_option = outside_option,
     dim_l_a  = dim_l_a,
     dim_u_a  = dim_u_a);

# Utility parameters
rand_service_coeffs = rand(P.S)
delta_s = normalize(kron(ones(P.K),rand_service_coeffs')); # for amentities
sigma_s = 1 ./ rand(P.S)
rand_loc_coeffs = rand(P.J)
delta_j = normalize(kron(rand_loc_coeffs,ones(P.K)'))

Delta_param = (delta_s = delta_s,
               sigma_s = sigma_s,
               delta_j = delta_j);

P = merge(P, Delta_param);

# Amenity parameters
c_a_j  = 1*ones(P.J)+0.5*rand(P.J); # what's the difference between c_a_j and c_s_j?
# should this be kappa?
c_s    = sigma_s; # σ_s???
w      = [8,6]; # wages???
lambda = 1;

Amenity_param = (c_a_j  = c_a_j,
                 c_s    = c_s,
                 w      = w,
                 lambda = lambda);

P = merge(P,Amenity_param);

# Supply parametersx - why are we taking random prices?
alpha = 1.2;
c = 0.5;
p = 0.3*minimum(P.w)*ones(P.J)+2*rand(P.J); # Any specific reason for these params?
r = rand(P.J);
a = rand(P.J,P.S); # matrix of amenities?

Supply_param = (alpha = alpha,
                c = c,
                p = p);

P = merge(P,Supply_param);

# Houses
H = vcat(fill.(0.8*sum(P.Pop)/P.J, P.J)...); # Why is this 0.8?

x = reshape([r a],P.J*(P.S+1));

# Check functions
@show Static_ED_vec(x,P)

# Save parameters
io = open("params/"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_only_wage_open.txt", "w")
write(io, string(P))
close(io)

#### Unconstrained optimization

# Check functions
@show Static_ED_vec(x,P)
@show Amenity_supply(x,P)
@show Static_EA(x,P)

# Define residual function
res_D(y) = Static_ED_vec(y,P)'*Static_ED_vec(y,P)/P.J;
res_A(y) = Static_EA(y,P)'*Static_EA(y,P)/(P.J*P.S);
res(y)   = res_D(exp.(y)) + res_A(exp.(y));


# Create an empty dataframe to store output
df = DataFrame()
for i in 1:(dim_l_a-1)
     colname = "r_$i"
     df[!,colname] = Float64[]
end
for i in dim_l_a:dim_u_a
     colname = "a_$(i+J)"
     df[!,colname] = Float64[]
end
df[!,"ED_vec_max"] = Float64[]
df[!,"EA_vec_max"] = Float64[]
df[!,"satisfies_constraints"] = Int64[]

# Loop over initial conditions
for i in 1:20

    println(i)

    initial_x = log.([i/4*ones(P.J); i*ones(P.J*P.S)])

    #results_NM = optimize(res,initial_x,iterations = 10^9, g_tol = 1e-12)
    #@show results_NM

    results = optimize(res,initial_x,method = LBFGS(); autodiff = :forward, iterations = 5*10^6)
    @show results
    io = open("optim_output/"*string(i/2)*"_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_only_wage_open.txt", "w")
    write(io, string(results))

    @show xmin = results.minimizer
    @show true_minimizer = exp.(xmin)
    @show ED_vec_max = maximum(abs.(Static_ED_vec(true_minimizer,P)))
    @show EA_vec_max = maximum(abs.(Static_EA(true_minimizer,P)))

    write(io,"ED_vec_max = $ED_vec_max\n")
    write(io,"EA_vec_max = $EA_vec_max\n")

    # If not converged properly, plug it into a Nelder-Mead optimizer
    if max(ED_vec_max,EA_vec_max) > 0.05
        results = optimize(res,xmin,method = NelderMead(); autodiff = :forward, iterations = 5*10^4)

        @show results
        io = open("optim_output/"*string(i/2)*"_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_NM_discrete_only_wage_open.txt", "w")
        write(io, string(results))

        @show xmin = results.minimizer
        @show true_minimizer = exp.(xmin)
        @show ED_vec_max = maximum(abs.(Static_ED_vec(true_minimizer,P)))
        @show EA_vec_max = maximum(abs.(Static_EA(true_minimizer,P)))

        write(io,"ED_vec_max = $ED_vec_max\n")
        write(io,"EA_vec_max = $EA_vec_max\n")
    end

    # Check if min of wages is higher than the max of rental prices
    if minimum(P.w) < maximum(true_minimizer[1:J])
        write(io,"Min wage < max rental price\n")
        append!(true_minimizer,ED_vec_max,EA_vec_max,0)
    else
        write(io,"Min wage >= max rental price\n")
        append!(true_minimizer,ED_vec_max,EA_vec_max,1)
    end
    close(io)

    push!(df, true_minimizer)
end

for i in 1:20

    println(i)

    initial_x = log.([i/4*ones(P.J); i*rand()*ones(P.J*P.S)])

    #results_NM = optimize(res,initial_x,iterations = 10^9, g_tol = 1e-12)
    #@show results_NM

    results = optimize(res,initial_x,method = LBFGS(); autodiff = :forward, iterations = 5*10^6)
    @show results
    io = open("optim_output/"*string(i/2)*"_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_only_wage_open.txt", "w")
    write(io, string(results))

    @show xmin = results.minimizer
    @show true_minimizer = exp.(xmin)
    @show ED_vec_max = maximum(abs.(Static_ED_vec(true_minimizer,P)))
    @show EA_vec_max = maximum(abs.(Static_EA(true_minimizer,P)))

    write(io,"ED_vec_max = $ED_vec_max\n")
    write(io,"EA_vec_max = $EA_vec_max\n")

    # If not converged properly, plug it into a Nelder-Mead optimizer
    if max(ED_vec_max,EA_vec_max) > 0.05
        results = optimize(res,xmin,method = NelderMead(); autodiff = :forward, iterations = 5*10^4)

        @show results
        io = open("optim_output/"*string(i/2)*"_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_NM_discrete_only_wage_open.txt", "w")
        write(io, string(results))

        @show xmin = results.minimizer
        @show true_minimizer = exp.(xmin)
        @show ED_vec_max = maximum(abs.(Static_ED_vec(true_minimizer,P)))
        @show EA_vec_max = maximum(abs.(Static_EA(true_minimizer,P)))

        write(io,"ED_vec_max = $ED_vec_max\n")
        write(io,"EA_vec_max = $EA_vec_max\n")
    end

    # Check if min of wages is higher than the max of rental prices
    if minimum(P.w) < maximum(true_minimizer[1:J])
        write(io,"Min wage < max rental price\n")
        append!(true_minimizer,ED_vec_max,EA_vec_max,0)
    else
        write(io,"Min wage >= max rental price\n")
        append!(true_minimizer,ED_vec_max,EA_vec_max,1)
    end
    close(io)

    push!(df, true_minimizer)
end

CSV.write("optim_output/"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_only_wage_open.csv", df)
