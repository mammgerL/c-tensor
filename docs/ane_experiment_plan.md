# ANE Experimental Plan (Non-Production Branch)

## Scope

This document defines a safe path to evaluate ANE backend feasibility without impacting the main Accelerate/OpenMP training path.

Target branch name:
- `experimental/ane-backend`

Non-goals:
- Do not replace default backend in `main`
- Do not claim production support

## Why experimental only

The referenced ANE approach (`maderix/ANE`) is research-oriented and relies on private APIs (`_ANEClient`, `_ANECompiler`).

Observed limitations from upstream notes:
- Incomplete training offload (parts still on CPU)
- Lower-than-expected ANE utilization on training workloads
- Stability/resource caveats around repeated compile/load cycles

## Milestones

1. `M0`: Keep current baseline stable
- Preserve current behavior:
  - macOS: Accelerate
  - Linux: OpenMP
- Freeze baseline metrics using `scripts/bench_backends.sh`

2. `M1`: ANE forward-only proof of concept
- Add separate backend files (`ane_backend.m`, `ane_backend.h`)
- Offload forward `matmul + add_bias + relu`
- Keep backward on current CPU path
- Compare forward latency only

3. `M2`: Hybrid training prototype
- Try hybrid graph execution:
  - Forward on ANE
  - Backward/update on CPU
- Measure end-to-end step latency
- Verify numerical drift vs baseline

4. `M3`: Decision gate
- Keep only if all are true:
  - stable over long runs
  - clear speedup over Accelerate baseline
  - maintainable build/runtime complexity
- Otherwise archive branch and keep Accelerate/OpenMP as final answer

## Metrics

For each build/backend, record:
- Train elapsed time (20,000 steps)
- Avg step latency
- Final eval accuracy
- Build complexity (extra deps, private APIs)
- Runtime stability (crashes/leaks over long runs)

## Risk controls

- Isolate ANE code path behind compile-time macro (e.g. `TENSOR_USE_ANE`)
- Never change default Makefile target to ANE
- Keep CI (if added later) on public APIs only

## Immediate next commands

```bash
# 1) baseline comparison (Accelerate vs OpenMP)
./scripts/bench_backends.sh

# 2) create experimental branch
git checkout -b experimental/ane-backend
```

## References

- ANE upstream repo: https://github.com/maderix/ANE
- Current project baseline docs: `README.md`
