# Resume HCL from d=30 batch 2 onwards
# Usage: ~/.juliaup/bin/julia +1.11.3 -t 12 --project scripts/experiments/run_HCL_remaining.jl

include(joinpath(@__DIR__, "HCL.jl"))

# d=30: batch 2 only (batch 1 already done)
@info "=== HCL d=30 batch 2 ==="
main(30, 1; dir="HCL", model_types=["network", "ridge", "lasso"])
GC.gc(true)

# d=40
@info "=== HCL d=40 batch 1 ==="
main(40, 1; dir="HCL", model_types=["pca_knn", "forest", "lgbm", "linear"])
GC.gc(true)

@info "=== HCL d=40 batch 2 ==="
main(40, 1; dir="HCL", model_types=["network", "ridge", "lasso"])
GC.gc(true)

# d=50: commented out, run separately after checking disk/results
# @info "=== HCL d=50 batch 1 ==="
# main(50, 1; dir="HCL", model_types=["pca_knn", "forest", "lgbm", "linear"])
# GC.gc(true)

# @info "=== HCL d=50 batch 2 ==="
# main(50, 1; dir="HCL", model_types=["network", "ridge", "lasso"])
# GC.gc(true)

@info "=== HCL d=30 and d=40 complete. Check results before continuing to d=50 ==="
