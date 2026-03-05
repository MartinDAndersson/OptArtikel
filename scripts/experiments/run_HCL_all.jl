# Run all HCL dimensions sequentially
# Usage: ~/.juliaup/bin/julia +1.11.3 -t 12 --project scripts/experiments/run_HCL_all.jl

include(joinpath(@__DIR__, "HCL.jl"))

# d=2: uses knn (analysis script treats d=2 differently)
@info "=== HCL d=2 ==="
main(2, 1; dir="carmona_dim", model_types=["knn", "network", "forest", "lgbm", "linear", "ridge", "lasso"])
GC.gc(true)

# d=10: all models at once
@info "=== HCL d=10 ==="
main(10, 1; dir="carmona_dim", model_types=["pca_knn", "network", "forest", "lgbm", "linear", "ridge", "lasso"])
GC.gc(true)

# d=20+: batch to avoid OOM
for d in [20, 30, 40, 50]
    @info "=== HCL d=$d batch 1 ==="
    main(d, 1; dir="carmona_dim", model_types=["pca_knn", "forest", "lgbm", "linear"])
    GC.gc(true)

    @info "=== HCL d=$d batch 2 ==="
    main(d, 1; dir="carmona_dim", model_types=["network", "ridge", "lasso"])
    GC.gc(true)
end

@info "=== All HCL experiments complete ==="
