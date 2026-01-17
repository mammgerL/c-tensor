#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static inline int argmax10(const float* row) {
    int best = 0; float bestv = row[0];
    for (int c = 1; c < 10; c++) {
        if (row[c] > bestv) { bestv = row[c]; best = c; }
    }
    return best;
}

// y 是负 onehot：正确类为 -1，其它为 0
static inline int label_from_neg_onehot10(const float* row) {
    int best = 0; float bestv = row[0];
    for (int c = 1; c < 10; c++) {
        if (row[c] < bestv) { bestv = row[c]; best = c; }
    }
    return best;
}

static void load_csv_n_fast(Tensor* x, Tensor* y, const char* filename, int N) {
    FILE* file = fopen(filename, "r");
    if (!file) { perror("Unable to open file"); exit(1); }

    static char file_buf[1 << 20];
    setvbuf(file, file_buf, _IOFBF, sizeof(file_buf));

    char line[20000];
    const int D = 784;
    const int C = 10;

    for (int b = 0; b < N; b++) {
        if (!fgets(line, (int)sizeof(line), file)) {
            fprintf(stderr, "Not enough rows in %s (got %d rows)\n", filename, b);
            break;
        }

        char* p = line;
        for (int i = 0; i < D + C; i++) {
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
                y->data->values[b * C + (i - D)] = v * (-1.0f);
            }

            p = end;
            while (*p == ',' || *p == ' ' || *p == '\t') p++;
        }
    }

    fclose(file);
}

int main() {
    // load model
    Tensor* w1 = NULL, * b1 = NULL, * w2 = NULL, * b2 = NULL;
    load_model("mnist_mlp.bin", &w1, &b1, &w2, &b2);

    // sanity checks
    if (!w1 || !b1 || !w2 || !b2) {
        fprintf(stderr, "load_model failed\n");
        return 1;
    }
    if (w1->data->ndim != 2 || w2->data->ndim != 2) {
        fprintf(stderr, "bad model: weights ndim\n");
        return 1;
    }
    if (w1->data->shape[0] != 784 || w2->data->shape[1] != 10) {
        fprintf(stderr, "bad model shapes: w1(%d,%d) w2(%d,%d)\n",
            w1->data->shape[0], w1->data->shape[1],
            w2->data->shape[0], w2->data->shape[1]);
        return 1;
    }
    int H = w1->data->shape[1];
    if (w2->data->shape[0] != H || b1->data->size != H || b2->data->size != 10) {
        fprintf(stderr, "bad model shapes: H=%d, w2(%d,%d), b1=%d, b2=%d\n",
            H, w2->data->shape[0], w2->data->shape[1], b1->data->size, b2->data->size);
        return 1;
    }

    // load test set
    const int N = 10000;
    Tensor* x = create_zero_tensor((int[]) { N, 784 }, 2);
    Tensor* y = create_zero_tensor((int[]) { N, 10 }, 2);
    load_csv_n_fast(x, y, "mnist_test.csv", N);

    // eval batch
    const int B = 256;
    Tensor* bx = create_zero_tensor((int[]) { B, 784 }, 2);
    Tensor* by = create_zero_tensor((int[]) { B, 10 }, 2);

    int correct = 0;
    int total = 0;

    const float* X = x->data->values;
    const float* Y = y->data->values;

    for (int start = 0; start < N; start += B) {
        int curB = (start + B <= N) ? B : (N - start);

        // pack batch
        for (int i = 0; i < curB; i++) {
            int idx = start + i;
            memcpy(bx->data->values + (size_t)i * 784,
                X + (size_t)idx * 784,
                (size_t)784 * sizeof(float));
            memcpy(by->data->values + (size_t)i * 10,
                Y + (size_t)idx * 10,
                (size_t)10 * sizeof(float));
        }

        // clear tail rows to avoid stale values (important for reproducibility/fair comparison)
        if (curB < B) {
            memset(bx->data->values + (size_t)curB * 784, 0,
                (size_t)(B - curB) * 784 * sizeof(float));
            memset(by->data->values + (size_t)curB * 10, 0,
                (size_t)(B - curB) * 10 * sizeof(float));
        }

        // forward using tensor.h ops (so backend matters)
        Tensor* h1 = matmul(bx, w1);      // (B,H)
        Tensor* h1b = add_bias(h1, b1);    // (B,H)
        Tensor* r1 = relu(h1b);           // (B,H)

        Tensor* h2 = matmul(r1, w2);      // (B,10)
        Tensor* h2b = add_bias(h2, b2);    // (B,10)

        // accuracy on first curB rows
        for (int i = 0; i < curB; i++) {
            int pred = argmax10(h2b->data->values + (size_t)i * 10);
            int gt = label_from_neg_onehot10(by->data->values + (size_t)i * 10);
            if (pred == gt) correct++;
            total++;
        }

        free_tensor(h1);
        free_tensor(h1b);
        free_tensor(r1);
        free_tensor(h2);
        free_tensor(h2b);
    }

    printf("Test Accuracy: %.2f%% (%d/%d)\n",
        100.0f * (float)correct / (float)total, correct, total);

    // cleanup
    free_tensor(w1); free_tensor(b1); free_tensor(w2); free_tensor(b2);
    free_tensor(x);  free_tensor(y);
    free_tensor(bx); free_tensor(by);

    return 0;
}
