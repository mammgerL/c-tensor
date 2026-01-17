#pragma once
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef TENSOR_FORCE_OPENMP
#define TENSOR_FORCE_OPENMP 0
#endif


#if defined(__APPLE__) && !TENSOR_FORCE_OPENMP
#define USE_ACCELERATE 1
#include <TargetConditionals.h>
#include <unistd.h>
#include <Accelerate/Accelerate.h>
#else
#define USE_ACCELERATE 0
#if defined(_OPENMP)
#include <omp.h>
#endif
#endif

#if defined(_OPENMP)
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define OMP_PARALLEL_FOR
#endif

#define MAX_PREVS 3
#define MAX_ARGS 5

// op codes
#define MATMUL     0
#define MEAN       1
#define MUL        2
#define RELU       3
#define LOGSOFTMAX 4
#define SUM_AXIS1  5
#define ADD_BIAS   6

typedef struct {
    float* values;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;

typedef union {
    int ival;
    float fval;
    int* ilist;
} Arg;

typedef struct Tensor {
    Arr* data;
    Arr* grad;
    int op;
    struct Tensor* prevs[MAX_PREVS];
    int num_prevs;
    Arg args[MAX_ARGS];
} Tensor;

// =========================
// random / init helpers
// =========================
static inline float random_normal() {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    if (u1 < 1e-12f) u1 = 1e-12f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}
static inline float rand_float() { return (float)rand() / (float)RAND_MAX; }
static inline float rand_range(float min, float max) { return min + rand_float() * (max - min); }

static inline float kaiming_uniform(int fan_in) {
    float gain = sqrtf(2.0f);
    float std = gain / sqrtf((float)fan_in);
    float bound = sqrtf(3.0f) * std;
    return rand_range(-bound, bound);
}

// =========================
// memory helpers (aligned + memset)
// =========================
static inline void* aligned_alloc_64(size_t bytes) {
    void* p = NULL;
#if defined(__APPLE__) || defined(__linux__)
    if (posix_memalign(&p, 64, bytes) != 0) p = NULL;
#else
    p = malloc(bytes);
#endif
    return p;
}

static inline Arr* create_arr_zeros(int* shape, int ndim) {
    Arr* arr = (Arr*)malloc(sizeof(Arr));
    if (!arr) return NULL;

    arr->ndim = ndim;
    arr->shape = (int*)malloc((size_t)ndim * sizeof(int));
    if (!arr->shape) { free(arr); return NULL; }
    memcpy(arr->shape, shape, (size_t)ndim * sizeof(int));

    arr->strides = (int*)malloc((size_t)ndim * sizeof(int));
    if (!arr->strides) { free(arr->shape); free(arr); return NULL; }

    arr->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        arr->strides[i] = arr->size;
        arr->size *= shape[i];
    }

    size_t bytes = (size_t)arr->size * sizeof(float);
    arr->values = (float*)aligned_alloc_64(bytes);
    if (!arr->values) {
        free(arr->strides);
        free(arr->shape);
        free(arr);
        return NULL;
    }
    memset(arr->values, 0, bytes);
    return arr;
}

static inline Arr* create_arr(float* data, int* shape, int ndim) {
    Arr* arr = create_arr_zeros(shape, ndim);
    if (!arr) return NULL;
    memcpy(arr->values, data, (size_t)arr->size * sizeof(float));
    return arr;
}

static inline void free_arr(Arr* a) {
    if (!a) return;
    free(a->values);
    free(a->shape);
    free(a->strides);
    free(a);
}

static inline Tensor* create_tensor(float* data, int* shape, int ndim) {
    Arr* d = create_arr(data, shape, ndim);
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = d;
    t->grad = create_arr_zeros(shape, ndim);
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

static inline Tensor* create_zero_tensor(int* shape, int ndim) {
    Arr* d = create_arr_zeros(shape, ndim);
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = d;
    t->grad = create_arr_zeros(shape, ndim);
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

static inline void free_tensor(Tensor* t) {
    if (!t) return;
    if (t->data) free_arr(t->data);
    if (t->grad) free_arr(t->grad);
    free(t);
}

// =========================
// debug
// =========================
static inline void print_tensor(Tensor* t) {
    printf("Tensor(\n");
    printf("\tdata: ");
    for (int i = 0; i < t->data->size; i++) printf("%f,", t->data->values[i]);
    printf("\n\tshape: ");
    for (int i = 0; i < t->data->ndim; i++) printf("%d,", t->data->shape[i]);
    printf("\n\tgrad: ");
    for (int i = 0; i < t->grad->size; i++) printf("%f,", t->grad->values[i]);
    printf("\n)\n");
}

// =========================
// forward/backward decls
// =========================
static inline Tensor* mul(Tensor* a, Tensor* b);
static inline void mul_backward(Tensor* out);

static inline Tensor* mean(Tensor* a);
static inline void mean_backward(Tensor* out);

static inline Tensor* sum_axis1(Tensor* inp);
static inline void sum_axis1_backward(Tensor* out);

static inline Tensor* matmul(Tensor* a, Tensor* b);
static inline void matmul_backward(Tensor* out);

static inline Tensor* logsoftmax(Tensor* inp);
static inline void logsoftmax_backward(Tensor* out);

static inline Tensor* relu(Tensor* inp);
static inline void relu_backward(Tensor* out);

static inline Tensor* add_bias(Tensor* inp, Tensor* bias);
static inline void add_bias_backward(Tensor* out);

// =========================
// autograd dispatcher
// =========================
static inline void backward(Tensor* t) {
    if (t->op == MUL) mul_backward(t);
    else if (t->op == MEAN) mean_backward(t);
    else if (t->op == MATMUL) matmul_backward(t);
    else if (t->op == RELU) relu_backward(t);
    else if (t->op == LOGSOFTMAX) logsoftmax_backward(t);
    else if (t->op == SUM_AXIS1) sum_axis1_backward(t);
    else if (t->op == ADD_BIAS) add_bias_backward(t);

    for (int i = 0; i < t->num_prevs; i++) backward(t->prevs[i]);
}

// =========================
// ops
// =========================
static inline Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* t = create_zero_tensor(a->data->shape, a->data->ndim);
    for (int i = 0; i < a->data->size; i++) t->data->values[i] = a->data->values[i] * b->data->values[i];
    t->op = MUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

static inline void mul_backward(Tensor* out) {
    for (int i = 0; i < out->data->size; i++) {
        out->prevs[0]->grad->values[i] += out->grad->values[i] * out->prevs[1]->data->values[i];
        out->prevs[1]->grad->values[i] += out->grad->values[i] * out->prevs[0]->data->values[i];
    }
}

static inline Tensor* mean(Tensor* t) {
    Tensor* m = create_zero_tensor((int[]) { 1 }, 1);
    float s = 0.0f;
    for (int i = 0; i < t->data->size; i++) s += t->data->values[i];
    m->data->values[0] = s / (float)t->data->size;
    m->op = MEAN;
    m->num_prevs = 1;
    m->prevs[0] = t;
    return m;
}

static inline void mean_backward(Tensor* out) {
    Tensor* inp = out->prevs[0];
    float g = out->grad->values[0] / (float)inp->data->size;
    for (int i = 0; i < inp->grad->size; i++) inp->grad->values[i] += g;
}

static inline Tensor* sum_axis1(Tensor* inp) {
    int B = inp->data->shape[0];
    int C = inp->data->shape[1];
    Tensor* out = create_zero_tensor((int[]) { B, 1 }, 2);

    for (int b = 0; b < B; b++) {
        float s = 0.0f;
        int base = b * inp->data->strides[0];
        for (int c = 0; c < C; c++) s += inp->data->values[base + c * inp->data->strides[1]];
        out->data->values[b] = s;
    }

    out->op = SUM_AXIS1;
    out->num_prevs = 1;
    out->prevs[0] = inp;
    return out;
}

static inline void sum_axis1_backward(Tensor* out) {
    Tensor* inp = out->prevs[0];
    int B = inp->data->shape[0];
    int C = inp->data->shape[1];

    for (int b = 0; b < B; b++) {
        float g = out->grad->values[b];
        int base = b * inp->grad->strides[0];
        for (int c = 0; c < C; c++) inp->grad->values[base + c * inp->grad->strides[1]] += g;
    }
}

static inline Tensor* relu(Tensor* inp) {
    Tensor* t = create_zero_tensor(inp->data->shape, inp->data->ndim);
    for (int i = 0; i < inp->data->size; i++) t->data->values[i] = (inp->data->values[i] > 0) ? inp->data->values[i] : 0.0f;
    t->op = RELU;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

static inline void relu_backward(Tensor* out) {
    Tensor* inp = out->prevs[0];
    for (int i = 0; i < out->data->size; i++) {
        inp->grad->values[i] += (inp->data->values[i] > 0) ? out->grad->values[i] : 0.0f;
    }
}

static inline Tensor* logsoftmax(Tensor* inp) {
    Tensor* t = create_zero_tensor(inp->data->shape, inp->data->ndim);
    int B = inp->data->shape[0];
    int C = inp->data->shape[1];

    for (int b = 0; b < B; b++) {
        int base = b * inp->data->strides[0];

        float maxv = inp->data->values[base];
        for (int c = 1; c < C; c++) {
            float v = inp->data->values[base + c * inp->data->strides[1]];
            if (v > maxv) maxv = v;
        }

        float sumexp = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = inp->data->values[base + c * inp->data->strides[1]];
            sumexp += expf(v - maxv);
        }
        float lse = logf(sumexp);

        for (int c = 0; c < C; c++) {
            int pos = base + c * inp->data->strides[1];
            t->data->values[pos] = inp->data->values[pos] - maxv - lse;
        }
    }

    t->op = LOGSOFTMAX;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

static inline void logsoftmax_backward(Tensor* out) {
    Tensor* inp = out->prevs[0];
    int B = out->data->shape[0];
    int C = out->data->shape[1];

    for (int b = 0; b < B; b++) {
        float gradsum = 0.0f;
        for (int c = 0; c < C; c++) gradsum += out->grad->values[b * C + c];

        for (int c = 0; c < C; c++) {
            int pos = b * C + c;
            float soft = expf(out->data->values[pos]);
            inp->grad->values[pos] += out->grad->values[pos] - soft * gradsum;
        }
    }
}

static inline Tensor* add_bias(Tensor* inp, Tensor* bias) {
    int B = inp->data->shape[0];
    int C = inp->data->shape[1];
    Tensor* out = create_zero_tensor(inp->data->shape, inp->data->ndim);

    // 支持 bias ndim=1 或 (1,C)
    for (int b = 0; b < B; b++) {
        int base = b * inp->data->strides[0];
        for (int c = 0; c < C; c++) {
            int pos = base + c * inp->data->strides[1];
            out->data->values[pos] = inp->data->values[pos] + bias->data->values[c];
        }
    }

    out->op = ADD_BIAS;
    out->num_prevs = 2;
    out->prevs[0] = inp;
    out->prevs[1] = bias;
    return out;
}

static inline void add_bias_backward(Tensor* out) {
    Tensor* inp = out->prevs[0];
    Tensor* bias = out->prevs[1];

    int B = out->data->shape[0];
    int C = out->data->shape[1];

    for (int i = 0; i < out->grad->size; i++) inp->grad->values[i] += out->grad->values[i];

    for (int c = 0; c < C; c++) {
        float s = 0.0f;
        for (int b = 0; b < B; b++) s += out->grad->values[b * C + c];
        bias->grad->values[c] += s;
    }
}

// =========================
// matmul (mac: vDSP forward + cblas backward; linux: loops + omp)
// =========================
static inline int is_contiguous_2d_rowmajor(const Tensor* t) {
    return (t && t->data && t->data->ndim == 2 &&
        t->data->strides[1] == 1 &&
        t->data->strides[0] == t->data->shape[1]);
}

#if USE_ACCELERATE
static inline void accel_init_once(void) {
    static int inited = 0;
    if (!inited) {
        // Apple vecLib 推荐用环境变量控制线程数
        // 对于小 GEMM：1 线程通常更快（避免调度开销）
        setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
        setenv("VECLIB_NUM_THREADS", "1", 1);
        inited = 1;
    }
}
#endif

static inline Tensor* matmul(Tensor* a, Tensor* b) {
    int P = a->data->shape[0];
    int Q = a->data->shape[1];
    int R = b->data->shape[1];

    Tensor* t = create_zero_tensor((int[]) { P, R }, 2);

#if USE_ACCELERATE
    accel_init_once();
    if (is_contiguous_2d_rowmajor(a) && is_contiguous_2d_rowmajor(b)) {
        vDSP_mmul(a->data->values, 1,
            b->data->values, 1,
            t->data->values, 1,
            (vDSP_Length)P, (vDSP_Length)R, (vDSP_Length)Q);
    }
    else
#endif
    {
        OMP_PARALLEL_FOR
            for (int i = 0; i < P; i++) {
                for (int j = 0; j < R; j++) {
                    float tmp = 0.0f;
                    for (int k = 0; k < Q; k++) {
                        int pos_a = i * a->data->strides[0] + k * a->data->strides[1];
                        int pos_b = k * b->data->strides[0] + j * b->data->strides[1];
                        tmp += a->data->values[pos_a] * b->data->values[pos_b];
                    }
                    t->data->values[i * R + j] = tmp;
                }
            }
    }

    t->op = MATMUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

static inline void matmul_backward(Tensor* out) {
    Tensor* a = out->prevs[0];
    Tensor* b = out->prevs[1];

    int P = a->data->shape[0];
    int Q = a->data->shape[1];
    int R = b->data->shape[1];

#if USE_ACCELERATE
    accel_init_once();

    const int fast =
        is_contiguous_2d_rowmajor(a) &&
        is_contiguous_2d_rowmajor(b) &&
        is_contiguous_2d_rowmajor(out) &&
        (a->grad && a->grad->ndim == 2 && a->grad->strides[1] == 1 && a->grad->strides[0] == a->grad->shape[1]) &&
        (b->grad && b->grad->ndim == 2 && b->grad->strides[1] == 1 && b->grad->strides[0] == b->grad->shape[1]);

    if (fast) {
        // dA += dC * B^T
        cblas_sgemm(CblasRowMajor,
            CblasNoTrans, CblasTrans,
            P, Q, R,
            1.0f,
            out->grad->values, R,
            b->data->values, R,
            1.0f,
            a->grad->values, Q);

        // dB += A^T * dC
        cblas_sgemm(CblasRowMajor,
            CblasTrans, CblasNoTrans,
            Q, R, P,
            1.0f,
            a->data->values, Q,
            out->grad->values, R,
            1.0f,
            b->grad->values, R);
        return;
    }
#endif

    // fallback loops
    OMP_PARALLEL_FOR
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < Q; j++) {
                float tmp = 0.0f;
                for (int k = 0; k < R; k++) {
                    int pos_b = j * b->data->strides[0] + k * b->data->strides[1];
                    tmp += out->grad->values[i * R + k] * b->data->values[pos_b];
                }
                a->grad->values[i * Q + j] += tmp;
            }
        }

    OMP_PARALLEL_FOR
        for (int i = 0; i < Q; i++) {
            for (int j = 0; j < R; j++) {
                float tmp = 0.0f;
                for (int k = 0; k < P; k++) {
                    int pos_a = k * a->data->strides[0] + i * a->data->strides[1];
                    tmp += out->grad->values[k * R + j] * a->data->values[pos_a];
                }
                b->grad->values[i * R + j] += tmp;
            }
        }
}

// =========================
// model save/load (WITH BIAS)
// =========================
typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t w1_rows, w1_cols;
    int32_t b1_len;
    int32_t w2_rows, w2_cols;
    int32_t b2_len;
} ModelHeader;

static inline void save_model(const char* path, Tensor* w1, Tensor* b1, Tensor* w2, Tensor* b2) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("save_model fopen"); exit(1); }

    ModelHeader h;
    h.magic = 0x314C504D;   // 'MLP1'
    h.version = 1;
    h.w1_rows = (int32_t)w1->data->shape[0];
    h.w1_cols = (int32_t)w1->data->shape[1];
    h.b1_len = (int32_t)b1->data->size;
    h.w2_rows = (int32_t)w2->data->shape[0];
    h.w2_cols = (int32_t)w2->data->shape[1];
    h.b2_len = (int32_t)b2->data->size;

    fwrite(&h, sizeof(h), 1, f);
    fwrite(w1->data->values, sizeof(float), (size_t)w1->data->size, f);
    fwrite(b1->data->values, sizeof(float), (size_t)b1->data->size, f);
    fwrite(w2->data->values, sizeof(float), (size_t)w2->data->size, f);
    fwrite(b2->data->values, sizeof(float), (size_t)b2->data->size, f);
    fclose(f);
}

static inline void load_model(const char* path, Tensor** w1_out, Tensor** b1_out, Tensor** w2_out, Tensor** b2_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("load_model fopen"); exit(1); }

    ModelHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fprintf(stderr, "bad model file\n"); exit(1); }
    if (h.magic != 0x314C504D || h.version != 1) { fprintf(stderr, "bad model header\n"); exit(1); }

    Tensor* w1 = create_zero_tensor((int[]) { (int)h.w1_rows, (int)h.w1_cols }, 2);
    Tensor* b1 = create_zero_tensor((int[]) { (int)h.b1_len }, 1);
    Tensor* w2 = create_zero_tensor((int[]) { (int)h.w2_rows, (int)h.w2_cols }, 2);
    Tensor* b2 = create_zero_tensor((int[]) { (int)h.b2_len }, 1);

    if (fread(w1->data->values, sizeof(float), (size_t)w1->data->size, f) != (size_t)w1->data->size) { fprintf(stderr, "bad model file (w1)\n"); exit(1); }
    if (fread(b1->data->values, sizeof(float), (size_t)b1->data->size, f) != (size_t)b1->data->size) { fprintf(stderr, "bad model file (b1)\n"); exit(1); }
    if (fread(w2->data->values, sizeof(float), (size_t)w2->data->size, f) != (size_t)w2->data->size) { fprintf(stderr, "bad model file (w2)\n"); exit(1); }
    if (fread(b2->data->values, sizeof(float), (size_t)b2->data->size, f) != (size_t)b2->data->size) { fprintf(stderr, "bad model file (b2)\n"); exit(1); }

    fclose(f);
    *w1_out = w1; *b1_out = b1; *w2_out = w2; *b2_out = b2;
}
