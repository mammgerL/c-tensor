#pragma once

#define ANE_BACKEND_OK 0
#define ANE_BACKEND_FALLBACK 1
#define ANE_BACKEND_ERROR -1

int ane_backend_is_available(void);
const char* ane_backend_last_error(void);

/*
 Fused forward op for M1 PoC:
 out = relu(x * w + b)
 x:   [batch, in_dim]
 w:   [in_dim, out_dim]
 b:   [out_dim]
 out: [batch, out_dim]
*/
int ane_dense_relu_forward(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int batch,
    int in_dim,
    int out_dim
);
