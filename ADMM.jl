# ==============================================================================
#   EdgeExpansionConvADMM -- Code to cmpute the convexified lower bound on
#                            the edge expansion problem.
# ------------------------------------------------------------------------------
#   Copyright (C) 2024 Melanie Siebenhofer <melaniesi@edu.aau.at>
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see https://www.gnu.org/licenses/.
# ==============================================================================

using Dates
using Printf
using MKL # test if this worked with BLAS.get_config() 

using Test, Documenter
using LinearAlgebra
using JuMP
using HiGHS
using Gurobi


include("grevlex-grlex-polytopes.jl");
include("RudyGraphIO.jl")

#########################################################################################################
#        E X A M P L E                                                                                  #
#  params = Parameters(1.5, 0.7, 1.24, 1e-5, 3000)                                                      #
#  L = RudyGraphIO.laplacian_from_RudyFile("../../04_Data/graphs/graph-instances/network/swingers.dat") #
#  OR                                                                                                   #
#  L = grlex(10)                                                                                        #
#  run_admm(L, params, PRSM=true)                                                                      #
#########################################################################################################


struct Parameters
    beta::Float64
    gamma1::Float64
    gamma2::Float64
    epsilon::Float64
    max_iterations::Int
end



const GRB_ENV = Ref{Gurobi.Env}()
function __init__()
    const GRB_ENV[] = Gurobi.Env()
    return
end
__init__()

function initialize_matricesADMM(n)
    nb = big(n)
    a = sum(binomial(nb,big(k)) / k for k in 1:Int(floor(n/2))) / sum(binomial(nb,big(k)) for k in 1:Int(floor(n/2)))
    b = sum(binomial(nb,big(k)) * k for k in 1:Int(floor(n/2))) / sum(binomial(nb,big(k)) for k in 1:Int(floor(n/2)))
    a = Float64(a)
    b = Float64(b)
    Y = [1/(n*(n-1))*(b-1)*ones(n,n) 2/(n-1)*(1-b/n)*ones(n,n) (floor(n/2)/n-b/n)*ones(n) (b/n-1/n)*ones(n) 1/n*ones(n);
     zeros(n,n) (a+(b-2*n-1)/(n*(n - 1)))*ones(n,n) (n*floor(n/2)*a-floor(n/2)-1+b/n)*ones(n) (-a+(n+1)/n-b/n)*ones(n) (a-1/n)*ones(n);
     zeros(1,n)    zeros(1,n)     (floor(n/2)^2*a - 2*floor(n/2) + b)  (floor(n/2) - 1)*(1 - a)   floor(n/2)*a-1;
     zeros(1,n)    zeros(1,n)       0                                           a-2+b     a-1;
     zeros(1,n)    zeros(1,n)       0                                           0           a]
    for i = 1:n
        Y[i,i] = 1/n
        Y[i,n+i] = 0
        Y[n+i,n+i] = a - 1/n
    end
    Y = Symmetric(Y)
    #Y = Symmetric(zeros(2*n+3, 2*n+3))
    R = ones(n+1,n+1); R[end,end] = 1; #zeros((n+1), (n+1))
    S = zeros(size(Y))
    return Y, R, S
end


function run_admm(L, params::Parameters; PRSM=true)
    n = size(L,1)
    L_tilde = [1/2 * L     zeros(n, (n + 3))  ;
               zeros(n,n)  1/2 * L  zeros(n,3);
               zeros(3,(2 * n + 3))]
    normLtilde = 4/n
    L_tilde *= 1/normLtilde
    dim_Y = 2 * n + 3


    # parameters
    eps = params.epsilon
    gamma1 = params.gamma1
    gamma2 = params.gamma2
    beta = params.beta
    max_it = params.max_iterations
    @assert beta > 0
    if PRSM
        @assert -1 < gamma1 < 1 && 0 < gamma2 < (1 + sqrt(5))/2
        @assert gamma1 + gamma2 > 0
        @assert abs(gamma1) < 1 + gamma2 - gamma2^2
    else
        gamma1 = gamma2 = max(gamma1, gamma2)
        @assert 0 < gamma1 < (1 + sqrt(5)) / 2
    end

    V = vcat(Matrix(I,n,n), -Matrix(I,n,n), -ones(1,n), ones(1,n), zeros(1,n))
    V = hcat(V, vcat(zeros(n), ones(n), floor(n/2), -1, 1))
    V = Matrix(qr(V).Q)

    # initialize Y, R, S
    Y, R, S = initialize_matricesADMM(n) 
    VRVt = V * R * V'
    Y = copy(VRVt)
    Yold = copy(Y)
        
    counter_iterationstotal = 0
    continue_iterations = true
    counter_stagnation = 0

    lower_bound = missing
    primal_obj = missing
    primal_obj_old = 0
    dual_obj = missing

    # ADMM iterations
    println("   primal       dual       err_p_rel    err_d_rel  iteration   time_elapsed")
    start_time = now()
    while continue_iterations
        counter_iterationstotal += 1
        Y = projection_polyhedralY(VRVt - (1/beta .* (L_tilde + S)))
        if PRSM
            S += (gamma1 * beta) * (Y - VRVt)
        end
        U_R, d_R = projection_PSD_cone(V' * (Symmetric(Y + 1/beta * S) * V))
        VU_R = V * U_R
        VRVt = Symmetric(VU_R * diagm(d_R) * VU_R')  #VRVt = V * R * V'
        primal_residual = Y - VRVt
        S += (gamma2 * beta) * primal_residual
        
        primal_obj = dot(L_tilde, Y)
        if abs(primal_obj - primal_obj_old) < 1e-6
            counter_stagnation += 1
        end
        primal_obj_old = primal_obj
        dual_obj = dot(L_tilde, VRVt)
        dual_residual = beta * V' * (Symmetric(Yold - Y) * V) # times beta
        Yold = copy(Y)
        rel_primal_residual = symm_norm(primal_residual) / (sqrt(dim_Y) + max(symm_norm(Y), symm_norm(VRVt)))
        #rel_primal_residual = symm_norm(primal_residual) / (1 + symm_norm(Y))
        VtSV = Symmetric(V' * (S * V))
        rel_dual_residual = beta * symm_norm(Symmetric(dual_residual)) / (sqrt(n + 1) + symm_norm(VtSV))
        #rel_dual_residual = beta * symm_norm(Symmetric(dual_residual)) / (1 + symm_norm(Symmetric(S)))
        if (rel_primal_residual < eps && rel_dual_residual < eps) || counter_iterationstotal == max_it
            continue_iterations = false
            lower_bound = compute_safe_lowerbound(L_tilde, S, V, VtSV)
            R = U_R * diagm(d_R) * U_R'
        end
        if counter_iterationstotal == 1 || counter_iterationstotal % 100 == 0 || !continue_iterations
            time_elapsed_s = Dates.value(Millisecond(now() - start_time)) / 10^3 # time elapsed in seconds
            @printf("%11.5f  %11.5f  %10.7f   %10.7f   %8d   %10.2f s\n",
                    primal_obj*normLtilde, dual_obj*normLtilde, rel_primal_residual, rel_dual_residual,
                    counter_iterationstotal, time_elapsed_s)
        end
    end
    total_time = Dates.value(Millisecond(now() - start_time)) / 10^3 # time elapsed in seconds
    println("=============================================================================")
    @printf("DNN lower bound:           %14.5f\n", lower_bound*normLtilde)
    @printf("time:                      %12.3f s\n", total_time)
    @printf("iterations:                %8d\n", counter_iterationstotal)
    @printf("stagnations:               %8d\n", counter_stagnation)
    println("=============================================================================")
    results = Dict{String, Any}("DNN-lb" => lower_bound*normLtilde, "time-wc" => total_time, "iterations" => counter_iterationstotal)
    results["Y"] = Y; results["R"] = R; results["S"] = S;
    results["primal"] = primal_obj; results["dual"] = dual_obj;
    return results
end

function symm_norm(A::Symmetric)
    return sqrt(dot(A,A))
end

"""
    projection_standardsimplex(x)

Computes the projection of vector `x` onto the standard simplex.

"""
function projection_standardsimplex(x)
    n = length(x)
    perm = sortperm(x, rev=true)
    if x[perm[n]] + (1 - sum(x)) / n > 0
        return copy(x) .+ (1 - sum(x)) / n
    end
    j = 1
    uj = x[perm[j]]
    aj = 1 - uj
    while uj + aj / j > 0
        j += 1
        uj = x[perm[j]]
        aj -= uj
    end
    aj += uj
    j -= 1
    λ = aj / j # j ≥ 1
    y =copy(x)
    y[perm[1:j]] .+= λ
    y[perm[(j+1):end]] .= 0  
    return y  
end
@testset "projection-standardsimplex" begin
    function gurobi_proj_standardsimplex(x)
        n = length(x)
        model = Model(Gurobi.Optimizer)
        y = @variable(model, y[1:n] >= 0)
        @constraint(model, sum(y) == 1)
        @objective(model, Min, (x - y)' * (x - y))
        optimize!(model)
        return value.(y)
    end
    x = 1 .- 1.5 * rand(30)
    @test norm(gurobi_proj_standardsimplex(x) - projection_standardsimplex(x)) < 1e-6

    x = ones(10)
    @test norm(1/10 * x - projection_standardsimplex(x)) < 1e-6

    x = - ones(10)
    @test norm(-1/10 * x - projection_standardsimplex(x)) < 1e-6

    x = zeros(10)
    @test norm(1/10 * ones(10) - projection_standardsimplex(x)) < 1e-6

    x = [1, 0, 0, 0, 0, 0, 0]
    @test norm(x -projection_standardsimplex(x)) < 1e-6

    x = [0, 0, 0, 0, 0, 0, 1]
    @test norm(x -projection_standardsimplex(x)) < 1e-6

    x = [0.5, 0, 0, 0, 0, 0, 0.5]
    @test norm(x -projection_standardsimplex(x)) < 1e-6
end

"""
    projection_polyhedralY(M)

Compute the projection of a symmetric matrix `M`
onto the polyhedral set \$\\mathcal{Y}\$.

In more detail, \$\\mathcal{Y}\$ is the set of symmetric matrices
with entries ≥ 0, ....
"""
function projection_polyhedralY(M)
    Y = copy(M)
    dim_Y = size(Y,1)
    n = (dim_Y - 3) >> 1
    nhm1 = floor(n/2) - 1
    onemnhinv = 1 - 1/floor(n/2)
    v = (2 * Y[1:n,end] + diag(Y[1:n,1:n])) / 3
    y1 = projection_standardsimplex(v)
    Y[1:n,end] .= y1
    for j = 1:n
        for i = 1:(j - 1)
            if Y[i,j] < 0
                Y[i,j] = 0
            elseif Y[i,j] > 1
                Y[i,j] = 1
            end
        end
        Y[j,j] = y1[j]
    end
    for j = (n + 1):(2 * n)
        for i = 1:(j - n)
            if Y[i,j] < 0
                Y[i,j] = 0
            elseif Y[i,j] > 1
                Y[i,j] = 1
            end
        end
        Y[(j - n),j] = 0
        for i = (j - n + 1):(j - 1)
            if Y[i,j] < 0
                Y[i,j] = 0
            elseif Y[i,j] > 1
                Y[i,j] = 1
            end
        end
        Y[j,j] = (2 * Y[j,end] + Y[j,j])/3
        if Y[j,j] < 0
            Y[j,j] = 0
        elseif Y[j,j] > 1
            Y[j,j] = 1
        end
        Y[j,end] = Y[j,j]
    end
    j = 2 * n + 1
    for i = 1:2*n
        if Y[i,j] < 0
            Y[i,j] = 0
        elseif Y[i,j] > nhm1
            Y[i,j] = nhm1
        end
    end
    if Y[2n+1,2n+1] < 0
        Y[2n+1,2n+1] = 0
    elseif Y[2n+1,2n+1] > floor(n/2) * nhm1
        Y[2n+1,2n+1] = floor(n/2) * nhm1
    end
    j = 2 * n + 2
    for i = 1:2n
        if Y[i,j] < 0
            Y[i,j] = 0
        elseif Y[i,j] > onemnhinv
            Y[i,j] = onemnhinv
        end
    end
    for i = 2n+1:2n+2
        if Y[i,j] < 0
            Y[i,j] = 0
        elseif Y[i,j] > nhm1
            Y[i,j] = nhm1
        end
    end
    if Y[2n+1,end] < 0
        Y[2n+1,end] = 0
    elseif Y[2n+1,end] > nhm1
        Y[2n+1,end] = nhm1
    end
    if Y[2n+2,end] < 0
        Y[2n+2,end] = 0
    elseif Y[2n+2,end] > onemnhinv
        Y[2n+2,end] = onemnhinv
    end
    if Y[end,end] < 1/floor(n/2)
        Y[end,end] = 1/floor(n/2)
    elseif Y[end,end] > 1
        Y[end,end] = 1
    end
    return Symmetric(Y, :U)
end

function gurobi_proj_polyhedralY(X)
    dim_Y = size(X, 1)
    n = (dim_Y - 3) >> 1

    model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    set_silent(model)
    Yhat = @variable(model, Yhat[1:(dim_Y - 1),1:(dim_Y - 1)] >= 0, Symmetric)
    y = @variable(model, y[1:(dim_Y - 1)] >= 0)
    ρ = @variable(model, 1/floor(n/2) ≤ ρ ≤ 1)
    @constraint(model, [i = 1:(2 * n)], Yhat[i,i] == y[i])
    @constraint(model, sum(y[1:n]) == 1)
    #@constraint(model, y[1:n] + y[(n + 1):(2 * n)] .== ρ)
    #@constraint(model, sum(y[1:n]) + y[2 * n + 1] == ρ * floor(n/2))
    #@constraint(model, sum(y[(n + 1):(2 * n)]) - y[2 * n + 2] == ρ)
    @constraint(model, Yhat[1:2n,1:2n] .≤ 1)
    @constraint(model, Yhat[1:2n,2n+1] .≤ floor(n/2) - 1)
    @constraint(model, Yhat[2n+1,2n+1]  ≤ floor(n/2)^2 - floor(n/2))
    @constraint(model, Yhat[1:2n,2n+2] .≤ 1 - 1/floor(n/2))
    @constraint(model, Yhat[2n+1:2n+2,2n+2] .≤ floor(n/2) - 1)
    @constraint(model, y[2n+1] ≤ floor(n/2) - 1)
    @constraint(model, y[2n+2] ≤ 1 - 1/floor(n/2))
    @constraint(model, [i = 1:n], Yhat[i,(n + i)] == 0)
    Ytilde = [Yhat  y;
            y'    ρ]
    @objective(model, Min, dot(X-Ytilde, X-Ytilde))
    optimize!(model)
    return Symmetric(value.(Ytilde))
end
@testset "projection-polyhedral" begin
    n = 5
    dim_Y = 2 * n + 3
    for _ = 1:5
        X =  1 .- 2 * rand(dim_Y, dim_Y)
        X = 1/2 * (X + X')
        @test norm(projection_polyhedralY(X) - gurobi_proj_polyhedralY(X)) < 1e-5
    end
end 

"""
    projection_PSD_cone(M)

Compute the projection of the matrix `M`
onto the cone of positive semidefinite
matrices.

Returns `U, d`, the projection can be computed
as `U * diagm(d) * U'`.
"""
function projection_PSD_cone(M)
    ev, U = eigen(Symmetric(M))
    ind1 = findfirst(>(0), ev)
    if isnothing(ind1)
        return zeros(size(M,1),1), [0]
        return zeros(size(M))
    end
    idx = ind1:length(ev)
    U = U[:,idx]
    return U, ev[idx]
    # return Symmetric(U * diagm(ev[idx]) * U')
end

"""
    projection_PSD_cone2(M)

Compute the projection of the matrix `M`
onto the cone of positive semidefinite
matrices.
"""
function projection_PSD_cone2(M)
    ev, U = eigen(Symmetric(M))
    mp1 = length(ev)
    ind1 = findfirst(>=(0), ev)
    if isnothing(ind1)
        return zeros(size(M))
    end
    if ind1 > floor(mp1/2)
        idx = ind1:mp1
        U = U[:,idx]
        return Symmetric(U * diagm(ev[idx]) * U')
    else
        idx = 1:ind1-1
        U = U[:,idx]
        return Symmetric(M - U * diagm(ev[idx]) * U')
    end
end

"""
    projection_PSD_cone_trace(M, n)

Compute the projection of the matrix `M`
onto the cone of positive semidefinite
matrices with trace equals `α`.
"""
function projection_PSD_cone_trace(M, α)
    ev, U = eigen(Symmetric(M))

    # project vector of eigenvalues onto the
    # simplex Δ_α
    # do not set entries who change to 0
    # (that is the smallest entries 1:(index - 1))
    cum_sum = 0
    index = length(ev)
    for k = 1:length(ev)
        if cum_sum + ev[index] - α ≥ k * ev[index]
            index += 1
            ev[index:end] .-= (cum_sum - α) / (k - 1)            
            break
        else
            cum_sum += ev[index]
            index -= 1
        end
    end
    if index == 0
        index = 1
        ev[1:end] .-= (cum_sum - α) / length(ev)
    end

    idx = index:length(ev)
    U = U[:,idx]
    return U, ev[idx]
    return Symmetric(U * diagm(ev[idx]) * U')
end


function compute_safe_lowerbound(L_tilde, Sout, V, VtSoutV)
    dim_Y = size(L_tilde, 1)
    n = (dim_Y - 3) >> 1

    ev, W = eigen(Symmetric(VtSoutV))
    ind1 = findfirst(>(0), ev)
    if isnothing(ind1)
        VMVt = zeros(dim_Y,dim_Y)
    else
        idx = ind1:length(ev)
        W = nu[:,idx]
        VW = V * W
        VMVt = VW * diagm(ev[idx]) * VW'
    end

    model = Model(HiGHS.Optimizer)
    set_attribute(model, "output_flag", false)
    @variable(model, Y[1:dim_Y,1:dim_Y] >= 0, Symmetric)
    @constraint(model, tr(Y[1:n,1:n]) == 1)
    @constraint(model, [i = 1:n], Y[i,(n + i)] == 0)
    @constraint(model, [i = 1:(2 * n)], Y[i,i] == Y[i,dim_Y])
    @constraint(model, Y[1:2n,1:2n] .≤ 1)
    @constraint(model, Y[1:2n,2n+1] .≤ floor(n/2) - 1)
    @constraint(model, Y[2n+1,2n+1]  ≤ floor(n/2)^2 - floor(n/2))
    @constraint(model, Y[1:2n,2n+2] .≤ 1 - 1/floor(n/2))
    @constraint(model, Y[2n+1:2n+2,2n+2] .≤ floor(n/2) - 1)
    @constraint(model, Y[2n+1,end] ≤ floor(n/2) - 1)
    @constraint(model, Y[2n+2,end] ≤ 1 - 1/floor(n/2))
    @constraint(model, 1/floor(n/2) ≤ Y[end,end] ≤ 1)
    @objective(model, Min, dot(L_tilde + Sout - VMVt, Y))
    optimize!(model)
    return objective_value(model)

end

#=
lbs = []
for gamma1 in 0.85:0.05:0.95
    for gamma2 in 0.7:0.05:1.1
        if gamma1 + gamma2 > 0 && abs(gamma1) < 1 + gamma2 - gamma2^2
            params = Parameters(10, gamma1, gamma2, 1e-5, 1000)
            #res = run_admm(L, params, PRSM=true)
            push!(lbs, (gamma1, gamma2, res["DNN-lb"]))
        end
    end
end
sort!(lbs, by=x->x[3])
=#
