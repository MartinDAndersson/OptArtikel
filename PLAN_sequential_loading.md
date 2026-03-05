# Plan: Sequential model loading for HCL d=50 analysis

## Context
The `analyze_and_clean` function in `scripts/analysis/HCL_analyze_and_clean.jl` loads ALL models into memory at once via `OptSwitch.load_models`. For d=50, this would require ~50 GB — far exceeding available RAM (~16 GB free). The fix is to load each model one at a time, extract lightweight results (strategy matrix + accumulated values), then free the model before loading the next.

## Key insight
`prepare_strategy_analysis` calls `compute_strategy_data` which processes each model independently in a loop, producing per-model:
- `strategy`: Int matrix (181 × 200) ~290 KB
- `accumulated_values`: Float64 matrix (181 × 200) ~290 KB
- `mean_values`: vector (181,) ~1.4 KB
- `strategy_name`: string

All downstream functions (distances, plots, comparisons) only need these lightweight outputs, **not** the model objects themselves. So we can load → extract → free, one model at a time.

## Changes

### File: `scripts/analysis/HCL_analyze_and_clean.jl`
**Replace line 88** (`mods = OptSwitch.load_models(...)`) with a sequential loop that:
1. Loads one model at a time via `OptSwitch.load_models([single_path])`
2. Calls `compute_strategy_for_value` + `calculate_accumulated_value` to extract strategy/values
3. Stores the lightweight results in pre-allocated vectors
4. Sets model reference to `nothing` and calls `GC.gc()` to free memory

**Replace lines 100-103** (the `prepare_strategy_analysis` call) with a call that passes the pre-computed strategy data instead of model objects. Two options:

**Option A (minimal change):** Create a small wrapper struct that mimics the model interface but just holds the pre-computed strategy + values. Pass these wrappers to `prepare_strategy_analysis` as-is.

**Option B (cleaner):** Factor the second half of `prepare_strategy_analysis` (everything after `compute_strategy_data`) into a separate function that takes pre-computed strategies/values directly. Call it from the script.

**Recommended: Option B** — avoids fake wrapper objects; the split is natural.

### File: `src/plots_common.jl`
1. Export `compute_strategy_for_value` and `calculate_accumulated_value` (if not already exported)
2. Extract lines 583-626 of `prepare_strategy_analysis` (everything after `compute_strategy_data`) into a new function, e.g. `prepare_strategy_analysis_from_data(strategies, accumulated_values, mean_values, strategy_names, sample_paths, times, dt, J, payoff, cost; initial_mode=1)`
3. Have the original `prepare_strategy_analysis` call `compute_strategy_data` then delegate to the new function, preserving backward compatibility

### Revised `analyze_and_clean` flow (pseudocode):
```julia
# Pre-compute shared data (same as before)
trajs = OptSwitch.generate_paths(...)
sample_paths = trajs[:, 1:end, 1:n_sample]
optimal_value = calculate_value_function(...)
greedy_strategies = calculate_greedy_value_matrix(...)[2]
optimal_strategies = determine_optimal_strategy(...)

# Process models ONE AT A TIME
strategies = Vector{Matrix{Int}}()
accumulated_values = Vector{Matrix{Float64}}()
mean_values_list = Vector{Vector{Float64}}()
strategy_names = Vector{String}()

for path in m
    mod = OptSwitch.load_models([data_dir * "/" * path])[1]
    predicted, strat = compute_strategy_for_value(mod, sample_paths, times, cost_c, payoff_c, dt, 1, J)
    acc = calculate_accumulated_value(strat, sample_paths, times, payoff_c, cost_c, dt; initial_mode=1)
    push!(strategies, strat)
    push!(accumulated_values, acc)
    push!(mean_values_list, vec(mean(acc, dims=2)))
    push!(strategy_names, mod.name)
    mod = nothing; GC.gc()
end

# Add other_strategies (a posteriori, greedy) - these are already lightweight

# Call new function with pre-computed data
strat_analysis = prepare_strategy_analysis_from_data(
    strategies, accumulated_values, mean_values, strategy_names,
    sample_paths, times, dt, J, payoff_c, cost_c; initial_mode=1
)
```

## Memory estimate for d=50
- Trajectories: (50 × 181 × 20000) Float64 = ~1.4 GB (freed after slicing)
- Sample paths: (50 × 181 × 200) Float64 = ~14 MB
- Per-model peak: one model loaded (~9 GB max for forest) + sample_paths (14 MB)
- All strategy results for ~8 models: 8 × ~580 KB = ~5 MB
- **Peak usage: ~9 GB for largest single model** — fits comfortably

## Verification
1. Run for d=40 first (already completed) and compare CSV output to existing `carmona_summary_d40.csv` — values should be identical
2. Then run for d=50 and monitor memory with `free -h` during execution
3. Check that plots are generated in `plots/carmona_highdim/` and `data/carmona_dim/`
