cd("C:/Users/marek/OneDrive/Documents/Julia_files")

include("functions_discrete_choice_housing.jl")
include("utils.jl")

using Random, LinearAlgebra, Optim, ForwardDiff, BenchmarkTools, DataFrames
using Optim: converged, maximum, maximizer, minimizer, iterations
using CSV, Plots, LineSearches
using Plots.PlotMeasures

Random.seed!(1234);

# Number of groups, number of amenities, number of locations
K = 2
S = 2
J = 2

# Population
Pop = 30*J/K*ones(P.K)

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
delta_s = [0.8 0.2; 0.2 0.8]; # for amentities
loc_coeffs = normalize([i for i in 1:P.J],1)
#loc_coeffs = [1/(2*(P.J-5))*ones(P.J-5);0.1*ones(5)]
delta_j = kron(loc_coeffs,ones(P.K)')
delta_r = normalize(ones(P.K),1)
sigma_s = 2*ones(P.S)
alpha_k = 0.5
alpha_vec = alpha_k*ones(P.K)

Delta_param = (delta_s = delta_s,
               sigma_s = sigma_s,
               delta_j = delta_j,
               delta_r = delta_r,
               alpha_vec = alpha_vec);

P = merge(P, Delta_param);

# Amenity parameters
c_a_j  = ones(P.J)
c_s    = sigma_s
w      = [6,3]
lambda = 1

Amenity_param = (c_a_j  = c_a_j,
                 c_s    = c_s,
                 w      = w,
                 lambda = lambda)

P = merge(P,Amenity_param)

# Supply parameters
alpha = 1.2
c = 0.5
p = 1*ones(P.J)
r = rand(P.J)
a = ones(P.J,P.S)

Supply_param = (alpha = alpha,
                c = c,
                p = p)

P = merge(P,Supply_param)

# Houses
H = 0.8*vcat(fill.(sum(P.Pop)/P.J, P.J)...)
#H = ones(P.J)

x = reshape([r a],P.J*(P.S+1))


# Check functions
@show Static_ED_vec(x,P)

# Save parameters
io = open("params/"*string(J)*"_"*string(K)*"_"*string(S)*"_search_mult_eq.txt", "w")
write(io, string(P))
close(io)

#### Unconstrained optimization

# Check functions
@show Static_ED_vec(x,P)
@show Amenity_supply(x,P)
@show Static_EA(x,P)

# Define residual function
res_D(y) = Static_ED_vec(y,P)'Static_ED_vec(y,P)
res_A(y) = Static_EA(y,P)'Static_EA(y,P)
res_f(y)   = res_D(exp.(y)) + res_A(exp.(y))
res(y) = log.(1e-10 .+ res_f(y))

r = zeros(P.J)
x = reshape([r a],P.J*(P.S+1))

@show Static_ED_vec([zeros(P.J);ones(P.J*P.S)],P)
if all(>(0),Static_ED_vec([zeros(P.J);ones(P.J*P.S)],P))
        println("Sufficient condition satisfied")
else
        println("Sufficient condition not satisfied")
end

# Create an empty dataframe to store output
df = DataFrame()
for i in 1:P.J
    colname = "initial_r_$i"
    df[!,colname] = Float64[]
end
for i in 1:P.J
    colname = "r_$i"
    df[!,colname] = Float64[]
end
for i in 1:P.J
    colname = "ED_$i"
    df[!,colname] = Float64[]
end
for i in 1:(P.J*P.S)
    colname = "initial_a_$i"
    df[!,colname] = Float64[]
end
for i in 1:(P.J*P.S)
    colname = "a_$i"
    df[!,colname] = Float64[]
end
for i in 1:(P.J*P.S)
    colname = "EA_$i"
    df[!,colname] = Float64[]
end
df[!,"ED_vec_max"] = Float64[]
df[!,"EA_vec_max"] = Float64[]
df[!,"Algo"] = AbstractString[]
df[!,"Converged"] = Int64[]
df[!,"Iterations"] = Float64[]

counter = 0

for l in 1:P.J
    for own_r in [0.01,2.99,3.01,5.99]
        for others_r in [0.01,2.99,3.01,5.99]
            for own_a_factor in 1:25
                for others_a_factor in 1:25
                    for own_a_rat in 1:5
                        for others_a_rat in 1:5

                            counter += 1

                            # Compute the initial values
                            initial_r = [others_r*ones(l-1); own_r; others_r*ones(P.J-l)]
                            initial_a = [hcat(others_a_factor*3*ones(l-1),others_a_factor*3/others_a_rat*ones(l-1));
                                            own_a_factor*3 own_a_factor*3/own_a_rat; hcat(others_a_factor*3*ones(P.J-l),others_a_factor*3/others_a_rat*ones(P.J-l))]
                            initial_x = reshape([initial_r initial_a],P.J*(P.S+1))

                            # Start with Nelder Mead first
                            results_NM = optimize(res, initial_x, iterations = 5*10^4, x_tol = 1e-32, f_tol = 1e-32, g_tol = 1e-16)
                            @show results_NM
                            xmin_NM_new_initial = results_NM.minimizer

                            # Plug into LBFGS
                            results = optimize(res, xmin_NM_new_initial, method = LBFGS(;linesearch=LineSearches.HagerZhang()); autodiff = :forward,
                                      x_tol = 1e-32, f_tol = 1e-32, g_tol = 1e-32, iterations = 5*10^6)
                            @show results

                            # Write the optim output into a txt file
                            io = open("optim_output/"*string(counter)*string(J)*"_"*string(K)*"_"*string(S)*"_LBFGS_search_mult_eq.txt", "w")
                            write(io, string(results))

                            # analyze results
                            @show xmin = results.minimizer
                            @show true_minimizer = exp.(xmin)
                            @show ED_vec_max = maximum(abs.(Static_ED_vec(true_minimizer,P)))
                            @show ED_true_minimizer = Static_ED_vec(true_minimizer,P)
                            @show EA_vec_max = maximum(abs.(Static_EA(true_minimizer,P)))
                            @show EA_true_minimizer = Static_EA(true_minimizer,P)
                            write(io,"ED_vec_max = $ED_vec_max\n")
                            write(io,"EA_vec_max = $EA_vec_max\n")

                            # Prepare to output into the dataframe
                            r_true_minimizer = true_minimizer[1:P.J]
                            a_true_minimizer = true_minimizer[(P.J+1):end]

                            current_row = [initial_r; r_true_minimizer; ED_true_minimizer; reshape(initial_a,P.J*P.S); a_true_minimizer; EA_true_minimizer; ED_vec_max; EA_vec_max; "LBFGS"; Int(converged(results)); iterations(results)]
                            #Push to df
                            push!(df, current_row)

                            # If not converged, plug back into NM
                            if (ED_vec_max > 1e-5) | (EA_vec_max > 1e-5)
                                results_NM = optimize(res, xmin, iterations = 5*10^7, x_tol = 1e-32, f_tol = 1e-32, g_tol = 1e-16)
                                @show results_NM
                                @show xmin = results_NM.minimizer
                                @show true_minimizer = exp.(xmin)
                                @show ED_vec_max = maximum(abs.(Static_ED_vec(true_minimizer,P)))
                                @show Static_ED_vec(true_minimizer,P)
                                @show EA_vec_max = maximum(abs.(Static_EA(true_minimizer,P)))
                                @show Static_EA(true_minimizer,P)
                                write(io,"ED_vec_max = $ED_vec_max\n")
                                write(io,"EA_vec_max = $EA_vec_max\n")

                                # Prepare to output into the dataframe
                                r_true_minimizer = true_minimizer[1:P.J]
                                a_true_minimizer = true_minimizer[P.J:end]

                                current_row = [initial_r; r_true_minimizer; ED_true_minimizer;reshape(initial_a,P.J*P.S); a_true_minimizer; EA_true_minimizer;ED_vec_max; EA_vec_max; "NM"; Int(converged(results_NM)); iterations(results_NM)]
                                #Push to df
                                push!(df, current_row)
                            end

                            # Periodically save output
                            if counter%100 == 0
                                CSV.write("optim_output/"*string(J)*"_"*string(K)*"_"*string(S)*"_search_mult_eq.csv", df)
                            end
                        end
                    end
                end
            end
        end
    end
end

# Save output
CSV.write("optim_output/"*string(J)*"_"*string(K)*"_"*string(S)*"_search_mult_eq.csv", df)
