# Analyze individual HCL dimensions and delete models to free space
# Usage: ~/.juliaup/bin/julia +1.11.3 -t 12 --project scripts/analysis/HCL_analyze_and_clean.jl

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir())
push!(LOAD_PATH, scriptsdir("plotting"))
push!(LOAD_PATH, scriptsdir("plot_functions_article"))

using Revise, Pkg, JLD2, FileIO, Random, Distributions, LinearAlgebra
using Parameters, BenchmarkTools, Printf, BSON, TimerOutputs, StaticArrays
using ProgressMeter, DataFrames, MLJ, Glob, MLUtils, CairoMakie
using AlgebraOfGraphics, CairoMakie, DataFramesMeta, Makie
using Lux, ADTypes, MLUtils, Optimisers, Zygote, Random, Statistics, Printf
import SimpleChains: static
import OptSwitch
using CSV
using hcl_plots

rng = MersenneTwister()
Random.seed!(rng, 54321)

# --- Reuse setup_carmona_problem from HCL_article_plots.jl ---
function setup_carmona_problem(d)
    function get_parameters(d)
        N = 180; J = 3; K = 20000; L = 1
        t_start = 0.0f0; t_end = 0.25f0 |> Float32
        dt = (t_end - t_start) / N; p = ()
        return Dict("d"=>d,"J"=>J,"N"=>N,"dt"=>dt,"K"=>K,"L"=>L,
            "t_start"=>t_start,"t_end"=>t_end,"p"=>p,"experiment"=>"carmona_dim")
    end
    @unpack d,N,J,K,L,t_start,t_end,dt = get_parameters(d)

    lambda_poisson = 32; lambda_exp = 10
    kappa = ones(d) .* 2 .|> Float32; kappa[1] = 5
    x0 = ones(d) .* 6; x0[1] = 50; x0 = x0 .|> Float32
    mu = log.(x0)
    sigma = [j == i ? 1.0f0 : 0.0f0 for i in 1:d, j in 1:d] .* 0.24
    sigma[1,:] .= ones(d).*0.32; sigma[1,1] = 0.5
    sigma = permutedims(sigma) .|> Float32
    p = Dict("sigma"=>sigma,"mu"=>mu,"d"=>d,"kappa"=>kappa,
        "lambda_poisson"=>lambda_poisson,"lambda_exp"=>lambda_exp)

    function dispersion(u,p,t); p["sigma"]; end
    function drift(u,p,t); kappa .* u .* (p["mu"] .- log.(u)); end
    function jump(u,p,dt)
        d=p["d"]; deltaN = rand(Binomial(1, lambda_poisson * Float64(dt)))
        size = rand(Exponential(1/lambda_exp)); j = exp(size)^deltaN .|> Float32
        du = ones(d); du[1] = j; (du .- 1) .* u
    end

    RandomProcess = OptSwitch.JumpProcess(drift, dispersion, jump)
    mat = [0 0.438 0.876; 0 -0.438*7.5 -0.876*10] .|> Float32 |> permutedims
    b = [-1.f0, -1.1f0, -1.2f0]
    payoff_p = Dict("mat"=>mat,"b"=>b,"J"=>J,"d"=>d)

    function payoff(x,t,pp)
        x1=x[1]; x2=mean(x[2:end]); res=pp["mat"]*[x1,x2].+pp["b"]; SVector{pp["J"]}(res)
    end
    function cost(x,t,pp)
        J=pp["J"]; c=0.01f0; C=ones(J,J).*c.+0.001f0.|>Float32; C[diagind(C)].=0.0f0; SMatrix{J,J}(C)
    end
    payoff_c(x,t) = payoff(x,t,payoff_p)
    cost_c(x,t) = cost(x,t,payoff_p)
    payoffmodel = OptSwitch.PayOffModel(payoff_p, payoff_c, cost_c)

    return RandomProcess, payoffmodel, x0, N, J, K, L, t_start, t_end, dt, p
end

# --- Analyze one dimension, save CSV, delete models ---
function analyze_and_clean(d)
    @info "=== Analyzing d=$d ==="
    RandomProcess, payoffmodel, x0, N, J, K, L, t_start, t_end, dt, p = setup_carmona_problem(d)

    data_dir = datadir("HCL/machines")
    m = readdir(data_dir)
    m = filter(x -> occursin("d=$d"*"_", x), m)
    if d == 2
        m = filter(x -> occursin("20000", x), m)
    else
        m = filter(x -> occursin("20000", x), m)
        m = filter(x -> !occursin("algorithm=knn", x), m)
        m = filter(x -> !occursin("algorithm=network 1", x), m)
        m = filter(x -> !occursin("algorithm=network 3", x), m)
    end
    @info "Found $(length(m)) models for d=$d"

    mods = OptSwitch.load_models(data_dir * "/" .* m)
    n_sample = d >= 30 ? 200 : 1000
    x_init = repeat(x0, 1, K)
    trajs = OptSwitch.generate_paths(RandomProcess, x_init, 0.0f0, N, dt, p)
    sample_paths = trajs[:, 1:end, 1:n_sample]
    times = 1:181

    payoff_c = payoffmodel.f; cost_c = payoffmodel.c
    optimal_value = calculate_value_function(sample_paths, cost_c, payoff_c, times, dt, J)
    greedy_strategies = calculate_greedy_value_matrix(sample_paths, payoff_c, cost_c, times, dt, 1, J)[2]
    optimal_strategies = determine_optimal_strategy(optimal_value, sample_paths, cost_c, times, dt, J, 1)

    strat_analysis = prepare_strategy_analysis(
        mods, payoff_c, cost_c, sample_paths, times, dt, J,
        other_strategies=[(optimal_strategies, "a posteriori"), (greedy_strategies, "greedy")],
        initial_mode=1
    )

    dist = strat_analysis.strat_dist
    joined_df = leftjoin(dist.distances, strat_analysis.summary_df, on=:Strategy)
    joined_df = select(joined_df, Not([:Rank, :DifferenceFromOptimal]))
    joined_df[!, :dimension] .= d

    mkpath(datadir("carmona_dim"))
    CSV.write(datadir("carmona_dim", "carmona_summary_d$(d).csv"), joined_df)
    @info "Saved carmona_summary_d$(d).csv"

    res = plot_strategy_analysis(strat_analysis)
    mkpath(plotsdir("carmona_highdim"))
    save(plotsdir("carmona_highdim/carmona_dim__strategy_distribution_d=$d.pdf"), dist.figure)
    save(datadir("carmona_dim/carmona_switching_strategies_$d.pdf"), res[1])
    save(datadir("carmona_dim/carmona_strategy_performance_$d.pdf"), res[2])
    @info "Saved plots for d=$d"

    # Models kept for now — delete manually after verifying results
end

# Run for completed dimensions
for d in [40]
    analyze_and_clean(d)
    GC.gc(true)
end

@info "=== Done analyzing d=2,10,20,30. Models deleted. Ready for d=40 batch 2 and d=50 ==="
