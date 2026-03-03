# ANE Execution Report (experimental/ane-backend)

This document records what was executed on local machine and what was learned.

## Environment

- Host: Apple Silicon macOS 26.3 (Darwin 25.3.0)
- Branch: `experimental/ane-backend`
- Date: 2026-03-03

## Commit timeline

1. `b1cf34e` — Add ANE scaffold and benchmark workflow
2. `b17f8f0` — Add runtime probe scaffold and cache stats
3. `b5dc33c` — Fix class detection to match in-memory runtime APIs
4. `99cab68` — Implement real ANE dense forward bridge with cached kernels

## Key findings

1. Class detection mismatch was the first blocker.
- `_ANECompiler` is not required for the in-memory path used by upstream code.
- Required classes on this machine are:
  - `_ANEClient`
  - `_ANEInMemoryModelDescriptor`
  - `_ANEInMemoryModel`
  - `_ANERequest`
  - `_ANEIOSurfaceObject`

2. Real ANE forward path is now running in local `ane_backend`.
- Mode: dense on ANE, bias+ReLU on CPU.
- Trigger: `ANE_ENABLE_PRIVATE_API=1`.

3. Dynamic weight path remains incomplete in local backend.
- `ANE_DYNAMIC_WEIGHTS=1` currently fails with descriptor creation in this code path.
- Upstream dynamic matmul test does work locally (see below), so the machine is capable.

## Reproduced commands and outcomes

1. Baseline train backend benchmark (project script)
```bash
make bench
```
- Accelerate elapsed: `6.306540s`
- OpenMP elapsed: `148.761266s`
- Speedup: `23.59x` (OpenMP / Accelerate)

2. Local ANE PoC (default fallback mode)
```bash
./ane_poc 128 20
```
- `ANE backend available: no`
- CPU fallback path active

3. Local ANE PoC (real ANE path)
```bash
ANE_ENABLE_PRIVATE_API=1 ./ane_poc 128 20
```
- `ANE backend available: yes`
- `ANE status: ANE eval OK (dense on ANE, bias/relu on CPU)`
- cache stats show compile once + cache hits

4. Upstream ANE repository validation
```bash
cd /tmp/ANE_test_43759/training
./test_ane_advanced
./test_perf_stats
./test_dynamic_matmul
```
- In-memory ANE compile/load/eval works.
- Dynamic matmul test passes correctness and performance checks.

5. Local dynamic mode trial (in this repo)
```bash
ANE_ENABLE_PRIVATE_API=1 ANE_DYNAMIC_WEIGHTS=1 ./ane_poc 128 20
```
- Current outcome: `ANE descriptor creation failed`
- Falls back to CPU.

6. Hybrid ANE training run (in this repo)
```bash
ANE_ENABLE_PRIVATE_API=1 TENSOR_USE_ANE=1 TRAIN_STEPS=300 ./train_ane
```
- Training loop completed successfully.
- Model file `mnist_mlp.bin` saved normally.
- Follow-up eval after this short run: `84.18% (8418/10000)`.

7. Full 20k comparison run (in this repo)
```bash
# Baseline
make train && ./train
make eval && ./eval

# ANE experimental path
make train-ane
ANE_ENABLE_PRIVATE_API=1 TENSOR_USE_ANE=1 TRAIN_STEPS=20000 ./train_ane
./eval
```
- Baseline train elapsed: `6.238239s`
- Baseline eval accuracy: `95.66% (9566/10000)`
- ANE train elapsed: `248.139270s`
- ANE avg step latency at end: `12.406 ms`
- ANE eval accuracy: `95.66% (9566/10000)`
- Relative train speed: ANE path is about `39.8x` slower than Accelerate baseline.

8. Dynamic-weight fix and rerun (in this repo)
```bash
# key fix: dynamic descriptor uses empty weight map (@{}) instead of nil
ANE_ENABLE_PRIVATE_API=1 ANE_DYNAMIC_WEIGHTS=1 ./ane_poc 128 20

# full train/eval
ANE_ENABLE_PRIVATE_API=1 ANE_DYNAMIC_WEIGHTS=1 TENSOR_USE_ANE=1 TRAIN_STEPS=20000 ./train_ane
./eval
```
- `ane_poc` dynamic result:
  - `compile=1 cache_hit=39 fallback=0`
  - ANE forward path works with dynamic weights.
- Full 20k dynamic run:
  - train elapsed: `8.082938s`
  - avg step near end: `~0.403 ms`
  - ANE stats: `compile=1 cache_hit=19999 fallback=0`
  - eval accuracy: `95.67% (9567/10000)`
- Relative speed vs Accelerate baseline (`6.238239s`):
  - dynamic ANE path is about `1.30x` slower.

9. Incremental optimization comparison (dynamic path)
```bash
ANE_ENABLE_PRIVATE_API=1 ANE_DYNAMIC_WEIGHTS=1 TENSOR_USE_ANE=1 TRAIN_STEPS=20000 ./train_ane
./eval
```
- Baseline dynamic path (before this round): `8.082938s`, acc `95.67%`
- Step1 (remove intermediate reorder/copy buffers, direct IOSurface write/read):
  - `7.798082s`, acc `95.67%`
- Step2 attempt (fuse bias+relu into ANE graph with second input):
  - runtime regression, fell back to CPU (`fallback=20000`), `151.769923s`
  - rolled back to keep stable path
- Step3 (reuse ANE `r1` tensor view, remove per-step create/copy/free for `r1`):
  - `7.592195s`, acc `95.67%`
- Net gain of accepted optimizations:
  - from `8.082938s` to `7.592195s` (~`6.1%` faster)
  - final dynamic path is about `1.22x` slower than Accelerate baseline.

## Current decision

- Keep ANE path experimental and opt-in only.
- Prefer dynamic-weight mode for experiments:
  - `ANE_ENABLE_PRIVATE_API=1 ANE_DYNAMIC_WEIGHTS=1 TENSOR_USE_ANE=1`
- Default production recommendation remains Accelerate on macOS.

## Next process (short)

1. Create a dedicated `ane_dynamic_poc.c` by reusing upstream helper flow.
2. Validate dynamic descriptor/compile/eval there first.
3. Port the validated helper into `ane_backend.m`.
4. Gate with `ANE_DYNAMIC_WEIGHTS=1`.
5. Integrate into `train.c` behind explicit opt-in and measure:
   - step time
   - final accuracy
   - compile/cache/fallback counters
