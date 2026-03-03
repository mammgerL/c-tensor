#include "tensor.h"
#include "ane_backend.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static void load_batch_x_csv(Tensor* bx, const char* filename, int batch) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("open mnist csv");
        exit(1);
    }

    static char file_buf[1 << 20];
    setvbuf(file, file_buf, _IOFBF, sizeof(file_buf));

    char line[20000];
    for (int b = 0; b < batch; b++) {
        if (!fgets(line, (int)sizeof(line), file)) {
            fprintf(stderr, "not enough rows in %s for batch=%d\n", filename, batch);
            fclose(file);
            exit(1);
        }

        char* p = line;
        for (int i = 0; i < 784 + 10; i++) {
            while (*p == ' ' || *p == '\t') p++;
            char* end = NULL;
            float v = strtof(p, &end);
            if (end == p) {
                fprintf(stderr, "csv parse error at row=%d col=%d\n", b, i);
                fclose(file);
                exit(1);
            }
            if (i < 784) {
                bx->data->values[b * 784 + i] = v;
            }
            p = end;
            while (*p == ',' || *p == ' ' || *p == '\t') p++;
        }
    }

    fclose(file);
}

static void forward_baseline_l1(Tensor* bx, Tensor* w1, Tensor* b1, float* out) {
    Tensor* h1 = matmul(bx, w1);
    Tensor* h1b = add_bias(h1, b1);
    Tensor* r1 = relu(h1b);

    memcpy(out, r1->data->values, (size_t)r1->data->size * sizeof(float));

    free_tensor(h1);
    free_tensor(h1b);
    free_tensor(r1);
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    const int batch = (argc > 1) ? atoi(argv[1]) : 128;
    const int iters = (argc > 2) ? atoi(argv[2]) : 2000;
    const int warmup = 20;

    if (batch <= 0 || iters <= 0) {
        fprintf(stderr, "usage: ./ane_poc [batch>0] [iters>0]\n");
        return 1;
    }

    Tensor* w1 = NULL;
    Tensor* b1 = NULL;
    Tensor* w2 = NULL;
    Tensor* b2 = NULL;
    load_model("mnist_mlp.bin", &w1, &b1, &w2, &b2);
    if (!w1 || !b1 || !w2 || !b2) {
        fprintf(stderr, "load_model failed, run training first\n");
        return 1;
    }

    if (w1->data->shape[0] != 784) {
        fprintf(stderr, "unexpected model shape for w1\n");
        return 1;
    }
    const int in_dim = 784;
    const int out_dim = w1->data->shape[1];

    Tensor* bx = create_zero_tensor((int[]) { batch, in_dim }, 2);
    load_batch_x_csv(bx, "mnist_test.csv", batch);

    float* baseline_out = (float*)aligned_alloc_64((size_t)batch * out_dim * sizeof(float));
    float* ane_out = (float*)aligned_alloc_64((size_t)batch * out_dim * sizeof(float));
    if (!baseline_out || !ane_out) {
        perror("aligned_alloc_64");
        return 1;
    }

    for (int i = 0; i < warmup; i++) {
        forward_baseline_l1(bx, w1, b1, baseline_out);
        ane_dense_relu_forward(bx->data->values, w1->data->values, b1->data->values, ane_out,
            batch, in_dim, out_dim);
    }

    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        forward_baseline_l1(bx, w1, b1, baseline_out);
    }
    double t1 = now_ms();

    int ane_rc = ANE_BACKEND_OK;
    double t2 = now_ms();
    for (int i = 0; i < iters; i++) {
        ane_rc = ane_dense_relu_forward(
            bx->data->values, w1->data->values, b1->data->values, ane_out, batch, in_dim, out_dim
        );
        if (ane_rc == ANE_BACKEND_ERROR) {
            fprintf(stderr, "ane forward failed: %s\n", ane_backend_last_error());
            return 1;
        }
    }
    double t3 = now_ms();

    float diff = max_abs_diff(baseline_out, ane_out, batch * out_dim);
    double baseline_ms = (t1 - t0) / (double)iters;
    double ane_ms = (t3 - t2) / (double)iters;

    printf("ANE backend available: %s\n", ane_backend_is_available() ? "yes" : "no");
    printf("ANE status: %s\n", ane_backend_last_error());
    if (ane_rc == ANE_BACKEND_FALLBACK) {
        printf("ANE path mode: CPU fallback (M1 scaffold)\n");
    }

    printf("L1 forward benchmark: batch=%d in=%d out=%d iters=%d\n", batch, in_dim, out_dim, iters);
    printf("baseline (tensor ops) avg: %.4f ms/iter\n", baseline_ms);
    printf("ane api path avg:          %.4f ms/iter\n", ane_ms);
    printf("max abs diff: %.8f\n", diff);
    {
        int compile_count = 0;
        int cache_hit_count = 0;
        int fallback_count = 0;
        ane_backend_get_stats(&compile_count, &cache_hit_count, &fallback_count);
        printf("ane stats: compile=%d cache_hit=%d fallback=%d\n",
            compile_count, cache_hit_count, fallback_count);
    }

    free(baseline_out);
    free(ane_out);

    free_tensor(bx);
    free_tensor(w1);
    free_tensor(b1);
    free_tensor(w2);
    free_tensor(b2);
    return 0;
}
