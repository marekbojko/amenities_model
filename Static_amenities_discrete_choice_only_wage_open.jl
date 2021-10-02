#cd("C:/Users/marek/OneDrive/Documents/Julia_files")

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
J = 20;

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
for i in 1:16

    println(i)

    initial_x = log.([i/4*ones(P.J); i*2*ones(P.J*P.S)])

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

df = CSV.read("optim_output/"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_only_wage_open.csv", DataFrame)

# Find those entries where we have convergence
df_converged = filter(row -> max(row.ED_vec_max,row.EA_vec_max) <= 8*10^-2 , df)

# Transform data on prices and amenities into a matrix
M = Matrix(df_converged[!,1:dim_u_a])

# Check if there is a single fixed point
one_fixed_pt = all([approx_equal(M[i,:],M[1,:]) for i in 2:(size(M)[1])])
if one_fixed_pt
    println("Unique limit point")

    # compute demand for housing
    r_pt = M[10,1:P.J]
    a_pt = reshape(M[10,P.dim_l_a:P.dim_u_a],P.J,P.S)
    ind_D_L = Static_D_L_prob_w_outside(r_pt,a_pt,P)[1:end-1,:]
    total_D_L = Static_D_L_w_outside(r_pt,a_pt,P)[1:end-1,:]
    ind_D_L_sqft = total_D_L*inv(Diagonal(P.Pop))
    total_amenities_by_loc = a_pt*ones(P.S)

    # Compute the dissimilarity index based on location
    @show dissim_index_loc = dissimilarity_index(ind_D_L[:,1],ind_D_L[:,2])

    # Compute the dissimilarity index based on sq footage
    @show dissim_index_sqft = dissimilarity_index(total_D_L[:,1],total_D_L[:,2])

    # Create scatter plot with housing prices
    loc_housing_coeffs = delta_j[:,1]
    scatter(r_pt, ind_D_L_sqft, title = "Housing demand, J=$J, K=$K, S=$S; DI = $dissim_index_sqft",
            xlabel = "r",ylabel = "Individual demand for L-term housing", labels = ["w=$(P.w[1])" "w=$(P.w[2])"])
    savefig("optim_output/housing_demand_ind_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_choice_only_wage_open.png")

    scatter(r_pt, total_D_L, title = "Housing demand, J=$J, K=$K, S=$S; DI = $dissim_index_sqft",
            xlabel = "r",ylabel = "Total demand for L-term housing", labels = ["w=$(P.w[1])" "w=$(P.w[2])"])
    savefig("optim_output/housing_demand_total_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_choice_only_wage_open.png")

    scatter(r_pt, total_amenities_by_loc, title = "Eq amenities, J=$J, K=$K, S=$S",
            xlabel = "r", ylabel = "Total amenities", legend = false)
    savefig("optim_output/Eq_amenities_house_prices_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_choice_only_wage_open.png")

    # Plot a scatter plot of amenities and housing demand
    scatter(a_pt[:,1], ind_D_L_sqft, title = "Housing demand vs amenities, s=1, J=$J, K=$K, S=$S; DI = $dissim_index_sqft",
            xlabel = "A_{s=1}",ylabel = "Individual demand for L-term housing (sq ft)", labels = ["w=$(P.w[1])" "w=$(P.w[2])"])
    savefig("optim_output/housing_demand_ind_amenities_s=1_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_choice_only_wage_open.png")

    # Plot a scatter plot of amenities and housing demand
    scatter(a_pt[:,2], ind_D_L_sqft, title = "Housing demand vs amenities, s=2, J=$J, K=$K, S=$S; DI = $dissim_index_sqft",
            xlabel = "A_{s=2}",ylabel = "Individual demand for L-term housing (sq ft)", labels = ["w=$(P.w[1])" "w=$(P.w[2])"])
    savefig("optim_output/housing_demand_ind_amenities_s=2_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_choice_only_wage_open.png")

    # Plots with housing demand and amenities available
    sorted_D_L = total_D_L[sortperm(get_ordered_indices(r_pt)),:]
    plot(sorted_D_L, label = "left", legend=:topleft, right_margin = 15mm,
    xticks=([10*i for i in 0:10]), labels = ["w=$(P.w[1])" "w=$(P.w[2])"], xlabel = "Index of location", ylabel = "Total housing")
    plot!(twinx(),r_pt[sortperm(get_ordered_indices(r_pt))],color=:green,xticks=:none,label="r", ylabel = "Price of housing", right_margin = 15mm)
    savefig("optim_output/housing_demand_r_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_choice_only_wage_open.png")

    sorted_a_pt = a_pt[sortperm(get_ordered_indices(r_pt)),:]
    plot(sorted_a_pt, label = "left", legend=:topleft, right_margin = 10mm,
    xticks=([10*i for i in 0:10]), labels = ["s=1" "s=2"], xlabel = "Index of location", ylabel = "Total amenities")
    plot!(twinx(),r_pt[sortperm(get_ordered_indices(r_pt))],color=:green,xticks=:none,label="r", ylabel = "Price of housing", right_margin = 10mm)
    savefig("optim_output/eq_amenities_r_"*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_discrete_choice_only_wage_open.png")
else
    println("MULTIPLE LIMIT POINTS")
end
