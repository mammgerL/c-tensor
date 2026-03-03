# ANE Runtime Bridge TODO (M1 -> M2)

This file tracks what is still missing for real ANE execution in `ane_backend.m`.

## Current status

- `ane_backend_is_available()`:
  - checks `ANE_ENABLE_PRIVATE_API=1`
  - probes `AppleNeuralEngine.framework` presence via `dlopen`
- probes `_ANEClient` and `_ANECompiler` classes via Objective-C runtime
- `ane_dense_relu_forward()`:
  - currently always executes CPU fallback (`cblas_sgemm + relu`)
  - has shape cache/compile counters for bridge integration points

## Next implementation steps

1. Build a minimal compile+run path for one fused graph:
   - input `[B, 784]`, weight `[784, H]`, bias `[H]`
   - output `[B, H]` with ReLU
2. Fill cache entries with actual compiled program handles.
3. Add deterministic correctness check against baseline path:
   - max abs diff threshold
   - failover to CPU fallback when out of bound
4. Add resource management for repeated compile/load cycles:
   - cap recompiles
   - explicit teardown hooks

## Safety constraints

- Keep ANE path opt-in only (`ANE_ENABLE_PRIVATE_API=1`).
- Never change default backend selection in `main`.
- Always preserve CPU fallback for correctness and recoverability.
