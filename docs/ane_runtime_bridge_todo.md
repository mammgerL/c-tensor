# ANE Runtime Bridge TODO (M1 -> M2)

This file tracks what is still missing for real ANE execution in `ane_backend.m`.

## Current status

- `ane_backend_is_available()`:
  - checks `ANE_ENABLE_PRIVATE_API=1`
  - probes `AppleNeuralEngine.framework` presence via `dlopen`
- probes runtime classes via Objective-C runtime:
  - `_ANEClient`
  - `_ANEInMemoryModelDescriptor`
  - `_ANEInMemoryModel`
  - `_ANERequest`
  - `_ANEIOSurfaceObject`
- `ane_dense_relu_forward()`:
  - default path (`ANE_DYNAMIC_WEIGHTS` unset): real ANE execution works
    - compile/load/evaluate via `_ANEInMemoryModel*`
    - dense runs on ANE, `+bias + relu` runs on CPU
  - dynamic mode (`ANE_DYNAMIC_WEIGHTS=1`): currently fails at descriptor creation on this code path
  - CPU fallback remains active and is used on any ANE failure
  - cache/compile/fallback counters are exposed by `ane_backend_get_stats()`

## Next implementation steps

1. Resolve dynamic-weight descriptor failure in local `ane_backend` path.
   - align runtime helper flow with the upstream `test_dynamic_matmul` implementation
   - keep one-time compile per shape and per-step weight IO update
2. Add deterministic correctness checks against baseline:
   - max abs diff threshold
   - failover to CPU fallback when out of bound
3. Add resource management for repeated compile/load cycles:
   - cap recompiles
   - explicit teardown hooks
4. Integrate ANE dense path into training loop behind explicit opt-in.
   - keep backward/update on CPU first (hybrid mode)
   - track step time and final accuracy regression

## Safety constraints

- Keep ANE path opt-in only (`ANE_ENABLE_PRIVATE_API=1`).
- Never change default backend selection in `main`.
- Always preserve CPU fallback for correctness and recoverability.
