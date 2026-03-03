#pragma once

#define ANE_BACKEND_OK 0
#define ANE_BACKEND_FALLBACK 1
#define ANE_BACKEND_ERROR -1

int ane_backend_is_available(void);
const char* ane_backend_last_error(void);
void ane_backend_get_stats(int* compile_count, int* cache_hit_count, int* fallback_count);

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

/*
 Dense forward op:
 out = x * w + b
*/
int ane_dense_forward(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int batch,
    int in_dim,
    int out_dim
);

/*
 Experimental fused MLP forward:
 r1 = relu(x * w1 + b1)
 logits = r1 * w2 + b2
*/
int ane_mlp2_forward(
    const float* x,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    float* out_r1,
    float* out_logits,
    int batch,
    int in_dim,
    int hidden_dim,
    int out_dim
);
