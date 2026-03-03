#include "tensor.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#if USE_ANE_RUNTIME
#include "ane_backend.h"
#endif

static void get_time(struct timeval* t) {
    gettimeofday(t, NULL);
}

static int env_int_or(const char* name, int defv) {
    const char* s = getenv(name);
    if (!s || !*s) return defv;
    return atoi(s);
}

static float env_float_or(const char* name, float defv) {
    const char* s = getenv(name);
    if (!s || !*s) return defv;
    return strtof(s, NULL);
}

static Tensor* create_tensor_view(float* data, int* shape, int ndim) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    Arr* d = create_arr_zeros(shape, ndim);
    if (!d) {
        free(t);
        return NULL;
    }
    free(d->values);
    d->values = data;

    t->data = d;
    t->grad = create_arr_zeros(shape, ndim);
    if (!t->grad) {
        free(d->shape);
        free(d->strides);
        free(d);
        free(t);
        return NULL;
    }
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

static void free_tensor_view(Tensor* t) {
    if (!t) return;
    if (t->data) {
        free(t->data->shape);
        free(t->data->strides);
        free(t->data);
    }
    if (t->grad) free_arr(t->grad);
    free(t);
}

/*
 CSV 解析优化：
  - fgets 读一行
  - 用 strtof(ptr, &end) 解析 float，并用指针推进
  - 避免 strtok 的全局状态/重复扫描
  - 避免 atof 的额外开销
*/
static void load_csv(Tensor* x, Tensor* y, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) { perror("Unable to open file"); exit(1); }

    // 给文件流更大的缓冲，减少系统调用/锁开销
    // 1<<20 = 1MB
    static char file_buf[1 << 20];
    setvbuf(file, file_buf, _IOFBF, sizeof(file_buf));

    // 每行最多 784+10 个数 + 逗号，10000 基本够；不够可再加
    char line[10000];

    const int N = x->data->shape[0];
    const int D = 784;
    const int C = 10;

    for (int b = 0; b < N; b++) {
        if (!fgets(line, (int)sizeof(line), file)) {
            fprintf(stderr, "Not enough rows in %s (got %d rows)\n", filename, b);
            break;
        }

        char* p = line;
        for (int i = 0; i < D + C; i++) {
            // 跳过可能的空格
            while (*p == ' ' || *p == '\t') p++;

            char* end = NULL;
            float v = strtof(p, &end);
            if (end == p) {
                fprintf(stderr, "CSV parse error at row %d col %d near: %.32s\n", b, i, p);
                exit(1);
            }

            if (i < D) {
                x->data->values[b * D + i] = v;
            }
            else {
                // 标签 onehot 乘 -1：正确类是 -1，其它是 0
                y->data->values[b * C + (i - D)] = v * (-1.0f);
            }

            p = end;
            // 跳过分隔符（逗号）以及可能的空白
            while (*p == ',' || *p == ' ' || *p == '\t') p++;
        }
    }

    fclose(file);
}

/*
  更快的 batch 采样（无放回）：
  - indices = [0..N-1]
  - Fisher–Yates shuffle
  - 每次连续取 B 个
  - 用完就重洗（等价于按 epoch 采样）
*/
typedef struct {
    int N;
    int pos;
    int* idx;
} BatchSampler;

static void sampler_init(BatchSampler* s, int N, unsigned seed) {
    s->N = N;
    s->pos = 0;
    s->idx = (int*)malloc((size_t)N * sizeof(int));
    if (!s->idx) { perror("malloc"); exit(1); }

    for (int i = 0; i < N; i++) s->idx[i] = i;
    srand(seed);
}

static void sampler_shuffle(BatchSampler* s) {
    // Fisher–Yates
    for (int i = s->N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = s->idx[i];
        s->idx[i] = s->idx[j];
        s->idx[j] = tmp;
    }
    s->pos = 0;
}

static void sampler_free(BatchSampler* s) {
    free(s->idx);
    s->idx = NULL;
}

/*
  用 sampler 提供的 idx 直接 memcpy batch
*/
static void get_next_batch(BatchSampler* s,
    Tensor* batch_x, Tensor* batch_y,
    Tensor* x, Tensor* y, int B) {
    const int N = x->data->shape[0];
    if (B > N) { fprintf(stderr, "Batch too large\n"); exit(1); }

    // 不够就重洗（相当于新 epoch）
    if (s->pos + B > s->N) {
        sampler_shuffle(s);
    }

    const int D = 784;
    const int C = 10;

    float* bx = batch_x->data->values;
    float* by = batch_y->data->values;
    const float* X = x->data->values;
    const float* Y = y->data->values;

    int base = s->pos;
    for (int i = 0; i < B; i++) {
        int idx = s->idx[base + i];
        memcpy(bx + (size_t)i * D, X + (size_t)idx * D, (size_t)D * sizeof(float));
        memcpy(by + (size_t)i * C, Y + (size_t)idx * C, (size_t)C * sizeof(float));
    }
    s->pos += B;
}

int main() {
    // ---------------- load data ----------------
    int N = 60000;
    Tensor* x = create_zero_tensor((int[]) { N, 784 }, 2);
    Tensor* y = create_zero_tensor((int[]) { N, 10 }, 2);

    load_csv(x, y, "mnist_train.csv");
    printf("loaded csv\n");

    // ---------------- model: 784 -> 256 -> 10 with bias ----------------
    int H = 256;
    const int O = 10;
    const int O_PAD = 16;

    Tensor* w1 = create_zero_tensor((int[]) { 784, H }, 2);
    Tensor* b1 = create_zero_tensor((int[]) { H }, 1);

    Tensor* w2 = create_zero_tensor((int[]) { H, O }, 2);
    Tensor* b2 = create_zero_tensor((int[]) { O }, 1);

    // init weights
    for (int i = 0; i < w1->data->size; i++) w1->data->values[i] = kaiming_uniform(784);
    for (int i = 0; i < w2->data->size; i++) w2->data->values[i] = kaiming_uniform(H);

    // ---------------- training hyperparams ----------------
    int B = env_int_or("TRAIN_BATCH", 128);
    float lr = env_float_or("TRAIN_LR", 0.005f);
    int steps = env_int_or("TRAIN_STEPS", 20000);

    Tensor* batch_x = create_zero_tensor((int[]) { B, 784 }, 2);
    Tensor* batch_y = create_zero_tensor((int[]) { B, O }, 2);

#if USE_ANE_RUNTIME
    const int use_ane = env_int_or("TENSOR_USE_ANE", 0) ? 1 : 0;
    const int ane_layer2_cfg = use_ane ? env_int_or("TENSOR_USE_ANE_LAYER2", -1) : 0;
    const int ane_layer2_min_macs = env_int_or("TENSOR_ANE_LAYER2_MIN_MACS", 1000000);
    const long long layer2_macs = (long long)B * H * O;
    int use_ane_layer2 = 0;
    if (use_ane) {
        if (ane_layer2_cfg > 0) use_ane_layer2 = 1;      // force on
        else if (ane_layer2_cfg == 0) use_ane_layer2 = 0; // force off
        else use_ane_layer2 = (layer2_macs >= ane_layer2_min_macs) ? 1 : 0; // auto
    }
    float* ane_r1_buf = NULL;
    float* ane_logits_buf = NULL;
    float* ane_logits16_buf = NULL;
    float* ane_w2_pad = NULL;
    float* ane_b2_pad = NULL;
    float* ane_dr1_buf = NULL;
    float* ane_dz_buf = NULL;
    Tensor* ane_r1_view = NULL;
    Tensor* ane_logits_view = NULL;
    if (use_ane) {
        ane_r1_buf = (float*)aligned_alloc_64((size_t)B * H * sizeof(float));
        if (use_ane_layer2) {
            ane_logits_buf = (float*)aligned_alloc_64((size_t)B * O * sizeof(float));
            ane_logits16_buf = (float*)aligned_alloc_64((size_t)B * O_PAD * sizeof(float));
            ane_w2_pad = (float*)aligned_alloc_64((size_t)H * O_PAD * sizeof(float));
            ane_b2_pad = (float*)aligned_alloc_64((size_t)O_PAD * sizeof(float));
            ane_dr1_buf = (float*)aligned_alloc_64((size_t)B * H * sizeof(float));
        }
        ane_dz_buf = (float*)aligned_alloc_64((size_t)B * H * sizeof(float));
        if (!ane_r1_buf || !ane_dz_buf ||
            (use_ane_layer2 && (!ane_logits_buf || !ane_logits16_buf || !ane_w2_pad || !ane_b2_pad || !ane_dr1_buf))) {
            perror("aligned_alloc_64");
            exit(1);
        }
        ane_r1_view = create_tensor_view(ane_r1_buf, (int[]) { B, H }, 2);
        if (!ane_r1_view) {
            perror("create_tensor_view");
            exit(1);
        }
        if (use_ane_layer2) {
            ane_logits_view = create_tensor_view(ane_logits_buf, (int[]) { B, O }, 2);
            if (!ane_logits_view) {
                perror("create_tensor_view");
                exit(1);
            }
            memset(ane_w2_pad, 0, (size_t)H * O_PAD * sizeof(float));
            memset(ane_b2_pad, 0, (size_t)O_PAD * sizeof(float));
        }
        if (ane_layer2_cfg < 0) {
            printf("ANE layer2 policy: auto (macs=%lld, threshold=%d) -> %s\n",
                layer2_macs, ane_layer2_min_macs, use_ane_layer2 ? "ANE" : "CPU");
        } else {
            printf("ANE layer2 policy: forced %s\n", use_ane_layer2 ? "ANE" : "CPU");
        }
        if (use_ane_layer2) {
            printf("ANE training mode enabled (dense1+dense2 forward on ANE)\n");
        } else {
            printf("ANE training mode enabled (dense1 forward on ANE)\n");
        }
    }
#endif

    // batch sampler
    BatchSampler sampler;
    sampler_init(&sampler, N, 0);
    sampler_shuffle(&sampler);

    struct timeval start, end;
    struct timeval t_last_print, t_step_start, t_step_end;

    get_time(&start);
    t_last_print = start;

    printf("Start Time: %ld.%06d seconds\n", (long)start.tv_sec, (int)start.tv_usec);

    const int PRINT_EVERY = 200;

    double last_step_ms = 0.0;
    int last_print_it = 0;

    // ---------------- training loop ----------------
    for (int it = 0; it < steps; it++) {
        get_time(&t_step_start);

        get_next_batch(&sampler, batch_x, batch_y, x, y, B);

        // forward
        Tensor* h1 = NULL;
        Tensor* h1b = NULL;
        Tensor* h2 = NULL;
        Tensor* h2b = NULL;
        Tensor* r1 = NULL;
        Tensor* logits = NULL;

#if USE_ANE_RUNTIME
        if (use_ane) {
            memset(ane_r1_view->grad->values, 0, (size_t)B * H * sizeof(float));
            if (use_ane_layer2) {
                memset(ane_logits_view->grad->values, 0, (size_t)B * O * sizeof(float));
                int rc = ane_dense_relu_forward(
                    batch_x->data->values,
                    w1->data->values,
                    b1->data->values,
                    ane_r1_buf,
                    B, 784, H
                );
                if (rc == ANE_BACKEND_ERROR) {
                    fprintf(stderr, "ANE forward error: %s\n", ane_backend_last_error());
                    return 1;
                }
                r1 = ane_r1_view;

                for (int h = 0; h < H; h++) {
                    memcpy(
                        ane_w2_pad + (size_t)h * O_PAD,
                        w2->data->values + (size_t)h * O,
                        (size_t)O * sizeof(float)
                    );
                }
                memcpy(ane_b2_pad, b2->data->values, (size_t)O * sizeof(float));

                rc = ane_dense_forward(
                    r1->data->values,
                    ane_w2_pad,
                    ane_b2_pad,
                    ane_logits16_buf,
                    B, H, O_PAD
                );
                if (rc == ANE_BACKEND_ERROR) {
                    fprintf(stderr, "ANE forward error: %s\n", ane_backend_last_error());
                    return 1;
                }
                for (int b = 0; b < B; b++) {
                    memcpy(
                        ane_logits_buf + (size_t)b * O,
                        ane_logits16_buf + (size_t)b * O_PAD,
                        (size_t)O * sizeof(float)
                    );
                }
                logits = ane_logits_view;
            } else {
                int rc = ane_dense_relu_forward(
                    batch_x->data->values,
                    w1->data->values,
                    b1->data->values,
                    ane_r1_buf,
                    B, 784, H
                );
                if (rc == ANE_BACKEND_ERROR) {
                    fprintf(stderr, "ANE forward error: %s\n", ane_backend_last_error());
                    return 1;
                }
                r1 = ane_r1_view;
                h2 = matmul(r1, w2);           // (B,10)
                h2b = add_bias(h2, b2);        // (B,10)
                logits = h2b;
            }
        } else
#endif
        {
            h1 = matmul(batch_x, w1);          // (B,H)
            h1b = add_bias(h1, b1);            // (B,H)
            r1 = relu(h1b);                    // (B,H)
            h2 = matmul(r1, w2);               // (B,10)
            h2b = add_bias(h2, b2);            // (B,10)
            logits = h2b;
        }

        Tensor* lout = logsoftmax(logits);      // (B,10)

        Tensor* mul_out = mul(lout, batch_y);  // (B,10)
        Tensor* per_sample = sum_axis1(mul_out); // (B,1)
        Tensor* loss = mean(per_sample);    // (1)

        // backward
        loss->grad->values[0] = 1.0f;
        backward(loss);

#if USE_ANE_RUNTIME
        if (use_ane) {
            const float* dr1 = NULL;
            if (use_ane_layer2) {
                const float* dlogits = ane_logits_view->grad->values;

                // dW2 += r1^T * dlogits
                cblas_sgemm(CblasRowMajor,
                    CblasTrans, CblasNoTrans,
                    H, O, B,
                    1.0f,
                    r1->data->values, H,
                    dlogits, O,
                    1.0f,
                    w2->grad->values, O);

                // db2 += sum_b dlogits[b,:]
                for (int o = 0; o < O; o++) {
                    float s = 0.0f;
                    for (int b = 0; b < B; b++) s += dlogits[b * O + o];
                    b2->grad->values[o] += s;
                }

                // dr1 = dlogits * W2^T
                cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasTrans,
                    B, H, O,
                    1.0f,
                    dlogits, O,
                    w2->data->values, O,
                    0.0f,
                    ane_dr1_buf, H);

                dr1 = ane_dr1_buf;
            } else {
                dr1 = r1->grad->values;
            }

            // manual backward for layer1: dz = d(r1) * relu'(z), z>0 <=> r1>0
            const float* r1v = r1->data->values;
            for (int i = 0; i < B * H; i++) {
                ane_dz_buf[i] = (r1v[i] > 0.0f) ? dr1[i] : 0.0f;
            }

            // dW1 += X^T * dZ
            cblas_sgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                784, H, B,
                1.0f,
                batch_x->data->values, 784,
                ane_dz_buf, H,
                1.0f,
                w1->grad->values, H);

            // db1 += sum_b dZ[b,:]
            for (int h = 0; h < H; h++) {
                float s = 0.0f;
                for (int b = 0; b < B; b++) s += ane_dz_buf[b * H + h];
                b1->grad->values[h] += s;
            }
        }
#endif

        // 在 free 之前把 loss 值存出来，避免 use-after-free
        float loss_val = loss->data->values[0];

        // SGD update
        {
            float* w = w1->data->values; float* g = w1->grad->values;
            int sz = w1->data->size;
            for (int i = 0; i < sz; i++) { w[i] -= g[i] * lr; g[i] = 0.0f; }
        }
        {
            float* w = b1->data->values; float* g = b1->grad->values;
            int sz = b1->data->size;
            for (int i = 0; i < sz; i++) { w[i] -= g[i] * lr; g[i] = 0.0f; }
        }
        {
            float* w = w2->data->values; float* g = w2->grad->values;
            int sz = w2->data->size;
            for (int i = 0; i < sz; i++) { w[i] -= g[i] * lr; g[i] = 0.0f; }
        }
        {
            float* w = b2->data->values; float* g = b2->grad->values;
            int sz = b2->data->size;
            for (int i = 0; i < sz; i++) { w[i] -= g[i] * lr; g[i] = 0.0f; }
        }

        // free intermediates
        if (h1) free_tensor(h1);
        if (h1b) free_tensor(h1b);
#if USE_ANE_RUNTIME
        if (!use_ane)
#endif
        free_tensor(r1);
        free_tensor(h2);
        free_tensor(h2b);
        free_tensor(lout);
        free_tensor(mul_out);
        free_tensor(per_sample);
        free_tensor(loss);

        get_time(&t_step_end);
        last_step_ms =
            (t_step_end.tv_sec - t_step_start.tv_sec) * 1000.0 +
            (t_step_end.tv_usec - t_step_start.tv_usec) / 1000.0;

        // print timing every PRINT_EVERY
        if ((it % PRINT_EVERY) == 0) {
            struct timeval now;
            get_time(&now);

            double window_ms =
                (now.tv_sec - t_last_print.tv_sec) * 1000.0 +
                (now.tv_usec - t_last_print.tv_usec) / 1000.0;

            int win_steps = it - last_print_it;
            if (win_steps <= 0) win_steps = 1;

            double avg_step_ms_in_window = window_ms / (double)win_steps;

            double since_start_ms =
                (now.tv_sec - start.tv_sec) * 1000.0 +
                (now.tv_usec - start.tv_usec) / 1000.0;

            double avg_step_ms_total = since_start_ms / (double)(it + 1);

            printf("batch: %d loss: %.6f | step: %.3f ms | avg(win): %.3f ms | avg(all): %.3f ms\n",
                it,
                loss_val,
                last_step_ms,
                avg_step_ms_in_window,
                avg_step_ms_total);

            t_last_print = now;
            last_print_it = it;
        }
    }

    get_time(&end);
    printf("End Time:   %ld.%06d seconds\n", (long)end.tv_sec, (int)end.tv_usec);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed Time: %.6f seconds\n", elapsed);

#if USE_ANE_RUNTIME
    if (use_ane) {
        int compile_count = 0;
        int cache_hit_count = 0;
        int fallback_count = 0;
        ane_backend_get_stats(&compile_count, &cache_hit_count, &fallback_count);
        printf("ANE stats: compile=%d cache_hit=%d fallback=%d\n",
            compile_count, cache_hit_count, fallback_count);
    }
#endif

    save_model("mnist_mlp.bin", w1, b1, w2, b2);
    printf("Saved model to mnist_mlp.bin\n");

    // free
    sampler_free(&sampler);

    free_tensor(x);
    free_tensor(y);
    free_tensor(batch_x);
    free_tensor(batch_y);

    free_tensor(w1);
    free_tensor(b1);
    free_tensor(w2);
    free_tensor(b2);

#if USE_ANE_RUNTIME
    if (use_ane) {
        free_tensor_view(ane_r1_view);
        if (ane_logits_view) free_tensor_view(ane_logits_view);
        free(ane_r1_buf);
        free(ane_logits_buf);
        free(ane_logits16_buf);
        free(ane_w2_pad);
        free(ane_b2_pad);
        free(ane_dr1_buf);
        free(ane_dz_buf);
    }
#endif

    return 0;
}
