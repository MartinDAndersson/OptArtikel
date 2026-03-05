# Train d=40 batch 2 + d=50 both batches
# Usage: ~/.juliaup/bin/julia +1.11.3 -t 12 --project scripts/experiments/run_HCL_d40_d50.jl

include(joinpath(@__DIR__, "HCL.jl"))

# d=40: batch 2 only (batch 1 already done)
@info "=== HCL d=40 batch 2 ==="
main(40, 1; dir="HCL", model_types=["network", "ridge", "lasso"])
GC.gc(true)

# d=50: both batches
@info "=== HCL d=50 batch 1 ==="
main(50, 1; dir="HCL", model_types=["pca_knn", "forest", "lgbm", "linear"])
GC.gc(true)

@info "=== HCL d=50 batch 2 ==="
main(50, 1; dir="HCL", model_types=["network", "ridge", "lasso"])
GC.gc(true)

@info "=== d=40 and d=50 training complete ==="
