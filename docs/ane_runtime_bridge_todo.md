# ANE Runtime Bridge TODO (M1 -> M2)

This file tracks what is still missing for real ANE execution in `ane_backend.m`.

## Current status

- `ane_backend_is_available()`:
  - checks `ANE_ENABLE_PRIVATE_API=1`
  - probes `AppleNeuralEngine.framework` presence via `dlopen`
- `ane_dense_relu_forward()`:
  - currently always executes CPU fallback (`cblas_sgemm + relu`)

## Next implementation steps

1. Resolve Objective-C runtime symbols/classes used by private ANE APIs.
2. Build a minimal compile+run path for one fused graph:
   - input `[B, 784]`, weight `[784, H]`, bias `[H]`
   - output `[B, H]` with ReLU
3. Add reusable compiled graph cache keyed by `(B, in_dim, out_dim)`.
4. Add deterministic correctness check against baseline path:
   - max abs diff threshold
   - failover to CPU fallback when out of bound
5. Add resource management for repeated compile/load cycles:
   - cap recompiles
   - explicit teardown hooks

## Safety constraints

- Keep ANE path opt-in only (`ANE_ENABLE_PRIVATE_API=1`).
- Never change default backend selection in `main`.
- Always preserve CPU fallback for correctness and recoverability.
