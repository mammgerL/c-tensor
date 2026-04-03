#pragma once
#include "tensor.h"

// 前向传播追踪结构，记录中间激活值
typedef struct {
    float* pixels;      // 输入像素 (784)
    int true_label;     // 真实标签
    int predicted;      // 预测标签
    float confidence;   // 预测置信度（softmax 概率）
    float* hidden;      // 隐藏层激活 (H=256)，ReLU 之后
    float* pre_relu;    // 隐藏层激活 (H=256)，ReLU 之前（加偏置后）
    float* matmul1;     // 隐藏层 matmul 输出 (H=256)，加偏置之前
    float* output;      // 输出层 logsoftmax (10)
    float* pre_softmax; // 输出层原始分数 (10)，logsoftmax 之前（加偏置后）
    float* matmul2;     // 输出层 matmul 输出 (10)，加偏置之前
    int hidden_size;
    int output_size;
} ForwardTrace;

static inline ForwardTrace* create_forward_trace(int hidden_size, int output_size) {
    ForwardTrace* trace = (ForwardTrace*)malloc(sizeof(ForwardTrace));
    if (!trace) { perror("malloc trace"); exit(1); }

    trace->pixels = (float*)malloc(784 * sizeof(float));
    if (!trace->pixels) { perror("malloc pixels"); exit(1); }

    trace->hidden = (float*)malloc(hidden_size * sizeof(float));
    if (!trace->hidden) { perror("malloc hidden"); exit(1); }

    trace->pre_relu = (float*)malloc(hidden_size * sizeof(float));
    if (!trace->pre_relu) { perror("malloc pre_relu"); exit(1); }

    trace->matmul1 = (float*)malloc(hidden_size * sizeof(float));
    if (!trace->matmul1) { perror("malloc matmul1"); exit(1); }

    trace->output = (float*)malloc(output_size * sizeof(float));
    if (!trace->output) { perror("malloc output"); exit(1); }

    trace->pre_softmax = (float*)malloc(output_size * sizeof(float));
    if (!trace->pre_softmax) { perror("malloc pre_softmax"); exit(1); }

    trace->matmul2 = (float*)malloc(output_size * sizeof(float));
    if (!trace->matmul2) { perror("malloc matmul2"); exit(1); }

    trace->hidden_size = hidden_size;
    trace->output_size = output_size;
    trace->true_label = -1;
    trace->predicted = -1;
    trace->confidence = 0.0f;

    return trace;
}

static inline void free_forward_trace(ForwardTrace* trace) {
    if (!trace) return;
    free(trace->pixels);
    free(trace->hidden);
    free(trace->pre_relu);
    free(trace->matmul1);
    free(trace->output);
    free(trace->pre_softmax);
    free(trace->matmul2);
    free(trace);
}

static inline int argmax(const float* data, int size) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}

// softmax: 从 logsoftmax 值计算实际概率
static inline void exp_from_logsoftmax(float* probs, const float* logprobs, int size) {
    float max_log = logprobs[0];
    for (int i = 1; i < size; i++) {
        if (logprobs[i] > max_log) max_log = logprobs[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        probs[i] = expf(logprobs[i] - max_log);
        sum += probs[i];
    }

    for (int i = 0; i < size; i++) {
        probs[i] /= sum;
    }
}

// 前向传播（单样本）并记录激活值
static inline ForwardTrace* forward_single(
    const float* x_single,
    int y_true,
    Tensor* w1, Tensor* b1,
    Tensor* w2, Tensor* b2
) {
    int H = w1->data->shape[1];
    ForwardTrace* trace = create_forward_trace(H, 10);

    memcpy(trace->pixels, x_single, 784 * sizeof(float));
    trace->true_label = y_true;

    Tensor* x = create_zero_tensor((int[]) { 1, 784 }, 2);
    memcpy(x->data->values, x_single, 784 * sizeof(float));

    Tensor* h1 = matmul(x, w1);
    Tensor* h1b = add_bias(h1, b1);
    Tensor* r1 = relu(h1b);

    Tensor* h2 = matmul(r1, w2);
    Tensor* h2b = add_bias(h2, b2);
    Tensor* lout = logsoftmax(h2b);

    memcpy(trace->matmul1, h1->data->values, H * sizeof(float));
    memcpy(trace->pre_relu, h1b->data->values, H * sizeof(float));
    memcpy(trace->hidden, r1->data->values, H * sizeof(float));
    memcpy(trace->matmul2, h2->data->values, 10 * sizeof(float));
    memcpy(trace->pre_softmax, h2b->data->values, 10 * sizeof(float));
    memcpy(trace->output, lout->data->values, 10 * sizeof(float));

    trace->predicted = argmax(trace->output, 10);
    trace->confidence = expf(trace->output[trace->predicted]);

    free_tensor(x);
    free_tensor(h1);
    free_tensor(h1b);
    free_tensor(r1);
    free_tensor(h2);
    free_tensor(h2b);
    free_tensor(lout);

    return trace;
}

static inline void append_float(char** buf, size_t* remaining, float val) {
    int written = snprintf(*buf, *remaining, "%.6f", val);
    if (written > 0 && (size_t)written < *remaining) {
        *buf += written;
        *remaining -= (size_t)written;
    }
}

static inline void append_float_array(char** buf, size_t* remaining,
                                       const float* data, int size) {
    char** p = buf;
    size_t* rem = remaining;

    int len = snprintf(*p, *rem, "[");
    if (len > 0 && (size_t)len < *rem) {
        *p += len;
        *rem -= (size_t)len;
    }

    for (int i = 0; i < size; i++) {
        if (i > 0) {
            len = snprintf(*p, *rem, ",");
            if (len > 0 && (size_t)len < *rem) {
                *p += len;
                *rem -= (size_t)len;
            }
        }
        append_float(p, rem, data[i]);
    }

    len = snprintf(*p, *rem, "]");
    if (len > 0 && (size_t)len < *rem) {
        *p += len;
        *rem -= (size_t)len;
    }
}

static inline void json_escape_string(char* dst, const char* src, size_t dst_size) {
    size_t j = 0;
    for (size_t i = 0; src[i] && j < dst_size - 2; i++) {
        switch (src[i]) {
            case '"':  dst[j++] = '\\'; dst[j++] = '"'; break;
            case '\\': dst[j++] = '\\'; dst[j++] = '\\'; break;
            case '\b': dst[j++] = '\\'; dst[j++] = 'b'; break;
            case '\f': dst[j++] = '\\'; dst[j++] = 'f'; break;
            case '\n': dst[j++] = '\\'; dst[j++] = 'n'; break;
            case '\r': dst[j++] = '\\'; dst[j++] = 'r'; break;
            case '\t': dst[j++] = '\\'; dst[j++] = 't'; break;
            default:
                dst[j++] = src[i];
        }
    }
    dst[j] = '\0';
}
