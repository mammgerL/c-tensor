#include "tensor.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static void get_time(struct timeval* t) {
    gettimeofday(t, NULL);
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

    Tensor* w1 = create_zero_tensor((int[]) { 784, H }, 2);
    Tensor* b1 = create_zero_tensor((int[]) { H }, 1);

    Tensor* w2 = create_zero_tensor((int[]) { H, 10 }, 2);
    Tensor* b2 = create_zero_tensor((int[]) { 10 }, 1);

    // init weights
    for (int i = 0; i < w1->data->size; i++) w1->data->values[i] = kaiming_uniform(784);
    for (int i = 0; i < w2->data->size; i++) w2->data->values[i] = kaiming_uniform(H);

    // ---------------- training hyperparams ----------------
    int B = 128;
    float lr = 0.005f;
    int steps = 20000;

    Tensor* batch_x = create_zero_tensor((int[]) { B, 784 }, 2);
    Tensor* batch_y = create_zero_tensor((int[]) { B, 10 }, 2);

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
        Tensor* h1 = matmul(batch_x, w1);       // (B,H)
        Tensor* h1b = add_bias(h1, b1);          // (B,H)
        Tensor* r1 = relu(h1b);                 // (B,H)

        Tensor* h2 = matmul(r1, w2);            // (B,10)
        Tensor* h2b = add_bias(h2, b2);          // (B,10)

        Tensor* lout = logsoftmax(h2b);          // (B,10)

        Tensor* mul_out = mul(lout, batch_y);  // (B,10)
        Tensor* per_sample = sum_axis1(mul_out); // (B,1)
        Tensor* loss = mean(per_sample);    // (1)

        // backward
        loss->grad->values[0] = 1.0f;
        backward(loss);

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
        free_tensor(h1);
        free_tensor(h1b);
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

    return 0;
}
