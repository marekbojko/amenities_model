#cd("C:/Users/marek/OneDrive/Documents/Julia_files")

cd("/home/mbojko/amenities_model")

include("functions_discrete_choice_dynamic.jl")
include("utils.jl")

using Random, LinearAlgebra, Optim, ForwardDiff, BenchmarkTools, DataFrames
using Optim: converged, maximum, maximizer, minimizer, iterations
using CSV, Plots, Plots.PlotMeasures

Random.seed!(1234)

# Number of groups, number of amenities, number of locations
K = 2
S = 2
J = 10

# Population
Pop = 30*J*[1/2, 1/2]

# Basic Params
P = (K = K,
     S = S,
     J = J,
     D = J+2,
     Pop = Pop)

# Utility parameters
rand_coeffs_delta_a = normalize(rand(P.S),1)
ordered_coeffs = normalize([i for i in 1:P.J],1)
Util_param = (delta_w = normalize(ones(P.K),1),
                delta_p = normalize(ones(P.K),1),
                delta_r = normalize(ones(P.K),1),
                delta_a = [0.8 0.2; 0.2 0.8],
                delta_j = [median(ordered_coeffs);ordered_coeffs[2:end]],
                beta = 0,
                w_in_util = true)
P = merge(P, Util_param)

# Moving cost parameters
#dist_mat = generate_rand_pos_def_symmetric_mat_w_0_diag(P.J+1)
dist_mat = ones(P.J+1,P.J+1) - Diagonal(ones(P.J+1))
iota_vecs = [i*ones(Int64, P.J+1)' for i in 1:(P.J+1)]
iota = vcat(iota_vecs...)
iota = [iota; [i for i in 1:(P.J+1)]']
moving_cost_param = (m_0 = 0,
                     m_1 = 0.5,
                     dist_mat = dist_mat,
                     iota = iota)
P = merge(P, moving_cost_param)

# Amenity parameters
Amenity_param = (c_a_j = 1*ones(P.J),
                sigma_s = 2*ones(P.S),
                w = [6,3],
                lambda = 1)
P = merge(P,Amenity_param)

# Supply parameters
Supply_param = (alpha = 1.2,
                c = 0.5,
                p = ones(P.J),
                H = vcat(fill.(0.8*sum(P.Pop)/P.J, P.J)...))
P = merge(P,Supply_param)

# Simulation parameters
value_function_iteration_param = (tol = 10^(-16),
                                  max_iter = 10^6)
P = merge(P,value_function_iteration_param)

# Some initial values for checks
r = rand(P.J)
a = rand(P.J,P.S)

x = reshape([r a],P.J*(P.S+1))

# Check functions
@show res_func(x,P)

# Save parameters
io = open("params/"*string(J)*"_"*string(K)*"_"*string(S)*"dynamic_w_wage_no_beta_counterfactuals_discrete.txt", "w")
write(io, string(P))
close(io)

#### Unconstrained optimization

# Define the objective function
f_obj(y) = res_func(exp.(y),P)

# initial guess
initial_x = log.([5*ones(P.J); 5*ones(P.J*P.S)])

# Perform the minimization task - first use Nelder Mead for 5*10^3 iters and then switch to LBFGS
results_NM = optimize(f_obj, initial_x, iterations = 5*10^4, x_tol = 1e-32, f_tol = 1e-32, g_tol = 1e-16)
@show results_NM
x_min_NM = results_NM.minimizer

results = optimize(f_obj, x_min_NM, method = LBFGS(); autodiff = :forward, iterations = 5*10^6,
                        x_tol = 1e-32, f_tol = 1e-32, g_tol = 1e-32)
@show results
io = open("optim_output/"*string(J)*"_"*string(K)*"_"*string(S)*"_NM_dynamic_w_wage_no_beta_counterfactuals.txt", "w")
write(io, string(results))

@show xmin = results.minimizer
@show true_minimizer = exp.(xmin)
@show ED_vec_max = maximum(abs.(ED_L(true_minimizer,P)))
@show ED_vec_eq = ED_L(true_minimizer,P)
@show EA_vec_max = maximum(abs.(EA_S(true_minimizer,P)))
@show EA_vec_eq = EA_S(true_minimizer,P)

write(io,"ED_vec_max = $ED_vec_max\n")
write(io,"ED_vec_eq = $ED_vec_eq\n")
write(io,"EA_vec_max = $EA_vec_max\n")
write(io,"EA_vec_eq = $EA_vec_eq\n")
write(io,"true_minimizer = $true_minimizer\n")
close(io)


#=
#### Analysis of results

r_eq = true_minimizer[1:P.J]
a_eq = reshape(true_minimizer[P.J+1:end],P.J,P.S)
stationary_dist_types_eq = [stationary_dist_one_type(true_minimizer,k,P) for k in 1:P.K]
stationary_dist_types_eq = hcat(stationary_dist_types_eq...)
D_L_eq = D_L(true_minimizer,P)
total_amenities_by_eq = a_eq*ones(P.S)

# Compute the dissimilarity index based on location
@show dissim_index_loc = dissimilarity_index(stationary_dist_types_eq[:,1],
            stationary_dist_types_eq[:,2])

# Plots with housing demand and amenities available
sorted_D_L = stationary_dist_types_eq[sortperm(get_ordered_indices(P.delta_j)),:]
plot(P.delta_j,sorted_D_L, label = "left", legend=:topleft,right_margin = 10mm, labels = ["w=$(P.w[1])" "w=$(P.w[2])"])
plot!(twinx(),P.delta_j,r_eq[sortperm(get_ordered_indices(P.delta_j))],color=:green,xticks=:none,label="r",right_margin = 10mm)
savefig("optim_output/housing_demand_r_"*string(J)*"_"*string(K)*"_"*string(S)*"_NM_dynamic_w_wage_no_beta.png")

sorted_a_eq = a_eq[sortperm(get_ordered_indices(P.delta_j)),:]
plot(P.delta_j,sorted_a_eq, label = "left", legend=:topleft,right_margin = 10mm, labels = ["s=1" "s=2"])
plot!(twinx(),P.delta_j,r_eq[sortperm(get_ordered_indices(P.delta_j))],color=:green,xticks=:none,label="r",right_margin = 10mm)
savefig("optim_output/eq_amenities_r_"*string(J)*"_"*string(K)*"_"*string(S)*"_NM_dynamic_w_wage_no_beta.png")

# Compute and plot welfare
@show welfare_by_type = welfare_households(true_minimizer,P)
bar(welfare_by_type, legend = false, xticks = [1,2], title = "Welfare by type")
savefig("optim_output/welfare_"*string(J)*"_"*string(K)*"_"*string(S)*"_NM_dynamic_w_wage_no_beta.png")

@show welfare_landlords_by_loc = welfare_landlords(true_minimizer,P)
bar(welfare_landlords_by_loc, legend = false, xticks = 1:P.J, title = "Landlord welfare by location")
savefig("optim_output/welfare_landlords_"*string(J)*"_"*string(K)*"_"*string(S)*"_NM_dynamic_w_wage_no_beta.png")

plot(P.delta_j,sorted_D_L)
=#
