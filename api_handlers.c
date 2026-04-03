#include "tensor.h"
#include "tensor_web.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <ctype.h>

void send_response(int client_fd, int status_code, const char* status_text,
                        const char* content_type, const char* body);
int get_query_param(const char* query, const char* param, char* value, size_t value_len);

static Tensor* g_w1 = NULL, * g_b1 = NULL, * g_w2 = NULL, * g_b2 = NULL;
static Tensor* g_test_x = NULL, * g_test_y = NULL;
static const int TEST_SIZE = 10000;
static int* g_pred_cache = NULL;
static int* g_true_cache = NULL;
static unsigned char* g_correct_cache = NULL;
static int g_eval_cache_ready = 0;
static int g_eval_correct_total = 0;
static int g_eval_total_by_class[10];
static int g_eval_correct_by_class[10];

extern void send_response(int client_fd, int status_code, const char* status_text,
                        const char* content_type, const char* body);
extern int get_query_param(const char* query, const char* param, char* value, size_t value_len);

typedef struct {
    int index;
    float activation;
    float weight;
    float contribution;
    float abs_contribution;
} HiddenContributor;

static void select_top_contributors(
    const float* hidden,
    const float* w2,
    int hidden_size,
    int class_idx,
    HiddenContributor* out,
    int k
) {
    for (int i = 0; i < k; i++) {
        out[i].index = -1;
        out[i].activation = 0.0f;
        out[i].weight = 0.0f;
        out[i].contribution = 0.0f;
        out[i].abs_contribution = -1.0f;
    }

    for (int h = 0; h < hidden_size; h++) {
        float a = hidden[h];
        float w = w2[h * 10 + class_idx];
        float c = a * w;
        float ac = fabsf(c);

        int pos = -1;
        for (int i = 0; i < k; i++) {
            if (ac > out[i].abs_contribution) {
                pos = i;
                break;
            }
        }
        if (pos < 0) continue;

        for (int i = k - 1; i > pos; i--) out[i] = out[i - 1];
        out[pos].index = h;
        out[pos].activation = a;
        out[pos].weight = w;
        out[pos].contribution = c;
        out[pos].abs_contribution = ac;
    }
}

static void ensure_eval_cache(void) {
    if (g_eval_cache_ready) return;
    if (!g_test_x || !g_test_y || !g_w1 || !g_b1 || !g_w2 || !g_b2) return;

    memset(g_eval_total_by_class, 0, sizeof(g_eval_total_by_class));
    memset(g_eval_correct_by_class, 0, sizeof(g_eval_correct_by_class));
    g_eval_correct_total = 0;

    for (int idx = 0; idx < TEST_SIZE; idx++) {
        const float* x_data = g_test_x->data->values + (size_t)idx * 784;
        int true_label = -1;
        for (int c = 0; c < 10; c++) {
            if (g_test_y->data->values[idx * 10 + c] < 0) {
                true_label = c;
                break;
            }
        }

        ForwardTrace* trace = forward_single(x_data, true_label, g_w1, g_b1, g_w2, g_b2);
        int pred = trace->predicted;
        int correct = (pred == true_label) ? 1 : 0;

        g_true_cache[idx] = true_label;
        g_pred_cache[idx] = pred;
        g_correct_cache[idx] = (unsigned char)correct;

        if (true_label >= 0 && true_label < 10) {
            g_eval_total_by_class[true_label]++;
            if (correct) g_eval_correct_by_class[true_label]++;
        }
        if (correct) g_eval_correct_total++;

        free_forward_trace(trace);
    }

    g_eval_cache_ready = 1;
}

void web_init() {
    load_model("mnist_mlp.bin", &g_w1, &g_b1, &g_w2, &g_b2);

    g_test_x = create_zero_tensor((int[]) { TEST_SIZE, 784 }, 2);
    g_test_y = create_zero_tensor((int[]) { TEST_SIZE, 10 }, 2);

    g_pred_cache = (int*)malloc((size_t)TEST_SIZE * sizeof(int));
    g_true_cache = (int*)malloc((size_t)TEST_SIZE * sizeof(int));
    g_correct_cache = (unsigned char*)malloc((size_t)TEST_SIZE * sizeof(unsigned char));
    if (!g_pred_cache || !g_true_cache || !g_correct_cache) {
        perror("malloc eval cache");
        exit(1);
    }
    memset(g_pred_cache, 0, (size_t)TEST_SIZE * sizeof(int));
    memset(g_true_cache, 0, (size_t)TEST_SIZE * sizeof(int));
    memset(g_correct_cache, 0, (size_t)TEST_SIZE * sizeof(unsigned char));

    FILE* file = fopen("mnist_test.csv", "r");
    if (!file) {
        perror("Unable to open mnist_test.csv");
        exit(1);
    }

    static char file_buf[1 << 20];
    setvbuf(file, file_buf, _IOFBF, sizeof(file_buf));

    char line[20000];
    for (int b = 0; b < TEST_SIZE; b++) {
        if (!fgets(line, (int)sizeof(line), file)) {
            fprintf(stderr, "Not enough rows in mnist_test.csv (got %d rows)\n", b);
            break;
        }

        char* p = line;
        for (int i = 0; i < 784 + 10; i++) {
            while (*p == ' ' || *p == '\t') p++;

            char* end = NULL;
            float v = strtof(p, &end);
            if (end == p) {
                fprintf(stderr, "CSV parse error at row %d col %d near: %.32s\n", b, i, p);
                exit(1);
            }

            if (i < 784) {
                g_test_x->data->values[b * 784 + i] = v;
            } else {
                g_test_y->data->values[b * 10 + (i - 784)] = v * (-1.0f);
            }

            p = end;
            while (*p == ',' || *p == ' ' || *p == '\t') p++;
        }
    }
    fclose(file);

    printf("Web server initialized: loaded model and test data\n");
}

static void send_json_response(int client_fd, const char* json) {
    send_response(client_fd, 200, "OK", "application/json", json);
}

void handle_api_architecture(int client_fd) {
    if (!g_w1 || !g_b1 || !g_w2 || !g_b2) {
        send_json_response(client_fd, "{\"error\":\"model not loaded\"}");
        return;
    }

    char response[2048];
    int H = g_w1->data->shape[1];

    int len = snprintf(response, sizeof(response),
        "{"
        "\"input_size\":784,"
        "\"hidden_size\":%d,"
        "\"output_size\":10,"
        "\"activations\":[\"none\",\"relu\",\"logsoftmax\"]"
        "}", H);

    if (len > 0 && len < (int)sizeof(response)) {
        send_json_response(client_fd, response);
    } else {
        send_json_response(client_fd, "{\"error\":\"buffer too small\"}");
    }
}

void handle_api_predict(int client_fd, const char* query) {
    if (!g_w1 || !g_b1 || !g_w2 || !g_b2) {
        send_json_response(client_fd, "{\"error\":\"model not loaded\"}");
        return;
    }

    char index_str[32];
    if (!get_query_param(query, "index", index_str, sizeof(index_str))) {
        send_json_response(client_fd, "{\"error\":\"missing index parameter\"}");
        return;
    }

    int index = atoi(index_str);
    if (index < 0 || index >= TEST_SIZE) {
        char error[128];
        snprintf(error, sizeof(error), "{\"error\":\"index out of range [0,%d]\"}", TEST_SIZE - 1);
        send_json_response(client_fd, error);
        return;
    }

    const float* x_data = g_test_x->data->values + (size_t)index * 784;
    int true_label = -1;

    for (int i = 0; i < 10; i++) {
        if (g_test_y->data->values[index * 10 + i] < 0) {
            true_label = i;
            break;
        }
    }

    ForwardTrace* trace = forward_single(x_data, true_label, g_w1, g_b1, g_w2, g_b2);

    const int top_k = 8;
    HiddenContributor top_contributors[8];
    float probabilities[10];
    float logits[10];
    int H = g_w1->data->shape[1];

    exp_from_logsoftmax(probabilities, trace->output, 10);
    for (int out = 0; out < 10; out++) {
        float acc = g_b2->data->values[out];
        for (int h = 0; h < H; h++) {
            acc += trace->hidden[h] * g_w2->data->values[h * 10 + out];
        }
        logits[out] = acc;
    }

    int runner_up = (trace->predicted == 0) ? 1 : 0;
    for (int i = 0; i < 10; i++) {
        if (i == trace->predicted) continue;
        if (probabilities[i] > probabilities[runner_up]) {
            runner_up = i;
        }
    }
    float margin = probabilities[trace->predicted] - probabilities[runner_up];

    int active_hidden = 0;
    float hidden_l1 = 0.0f;
    float hidden_max = 0.0f;
    for (int h = 0; h < H; h++) {
        float v = trace->hidden[h];
        if (v > 0.0f) active_hidden++;
        hidden_l1 += fabsf(v);
        if (fabsf(v) > hidden_max) hidden_max = fabsf(v);
    }

    select_top_contributors(
        trace->hidden,
        g_w2->data->values,
        H,
        trace->predicted,
        top_contributors,
        top_k
    );

    char* buffer = (char*)malloc(196608);
    if (!buffer) {
        free_forward_trace(trace);
        send_json_response(client_fd, "{\"error\":\"out of memory\"}");
        return;
    }

    char* p = buffer;
    size_t remaining = 196608;
    int len = snprintf(p, remaining,
        "{"
        "\"index\":%d,"
        "\"true_label\":%d,"
        "\"predicted\":%d,"
        "\"correct\":%s,"
        "\"confidence\":%.6f,"
        "\"margin\":%.6f,"
        "\"runner_up\":%d,"
        "\"pixels\":",
        index, trace->true_label, trace->predicted,
        trace->predicted == trace->true_label ? "true" : "false",
        trace->confidence,
        margin,
        runner_up);
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }
    append_float_array(&p, &remaining, trace->pixels, 784);

    len = snprintf(p, remaining, ",\"hidden\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }
    append_float_array(&p, &remaining, trace->hidden, trace->hidden_size);

    len = snprintf(p, remaining, ",\"output\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }
    append_float_array(&p, &remaining, trace->output, trace->output_size);

    len = snprintf(p, remaining, ",\"probabilities\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }
    append_float_array(&p, &remaining, probabilities, 10);

    len = snprintf(p, remaining, ",\"logits\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }
    append_float_array(&p, &remaining, logits, 10);

    len = snprintf(
        p,
        remaining,
        ",\"learning\":{"
        "\"active_hidden\":%d,"
        "\"hidden_sparsity\":%.6f,"
        "\"hidden_l1\":%.6f,"
        "\"hidden_max_abs\":%.6f,"
        "\"top_contributors\":[",
        active_hidden,
        1.0f - ((float)active_hidden / (float)H),
        hidden_l1,
        hidden_max
    );
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    int appended = 0;
    for (int i = 0; i < top_k; i++) {
        if (top_contributors[i].index < 0) continue;
        len = snprintf(
            p,
            remaining,
            "%s{\"hidden_index\":%d,\"activation\":%.6f,\"weight_to_pred\":%.6f,\"contribution\":%.6f}",
            appended ? "," : "",
            top_contributors[i].index,
            top_contributors[i].activation,
            top_contributors[i].weight,
            top_contributors[i].contribution
        );
        if (len > 0 && (size_t)len < remaining) {
            p += len;
            remaining -= (size_t)len;
            appended = 1;
        }
    }

    len = snprintf(
        p,
        remaining,
        "]},\"computation\":{\"steps\":["
        "{\"name\":\"Layer 1: Input -> Hidden\",\"operation\":\"matmul + bias + relu\"},"
        "{\"name\":\"Layer 2: Hidden -> Output\",\"operation\":\"matmul + bias + logsoftmax\"}"
        "]}}"
    );
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    send_json_response(client_fd, buffer);
    free(buffer);
    free_forward_trace(trace);
}

void handle_api_eval(int client_fd) {
    if (!g_w1 || !g_b1 || !g_w2 || !g_b2) {
        send_json_response(client_fd, "{\"error\":\"model not loaded\"}");
        return;
    }

    ensure_eval_cache();
    if (!g_eval_cache_ready) {
        send_json_response(client_fd, "{\"error\":\"cache not ready\"}");
        return;
    }

    int total = TEST_SIZE;
    int correct = g_eval_correct_total;

    char response[4096];
    float accuracy = 100.0f * (float)correct / (float)total;

    int len = snprintf(response, sizeof(response),
        "{"
        "\"total\":%d,"
        "\"correct\":%d,"
        "\"accuracy\":%.2f,"
        "\"incorrect\":%d,"
        "\"cached\":true,"
        "\"by_class\":[",
        total, correct, accuracy, total - correct);

    for (int i = 0; i < 10; i++) {
        int class_len;
        if (i == 0) {
            class_len = snprintf(response + len, sizeof(response) - len,
                "{\"digit\":%d,\"correct\":%d,\"total\":%d,\"accuracy\":%.2f}",
                i, g_eval_correct_by_class[i], g_eval_total_by_class[i],
                g_eval_total_by_class[i] > 0
                    ? (100.0f * g_eval_correct_by_class[i]) / g_eval_total_by_class[i]
                    : 0.0f);
        } else {
            class_len = snprintf(response + len, sizeof(response) - len,
                ",{\"digit\":%d,\"correct\":%d,\"total\":%d,\"accuracy\":%.2f}",
                i, g_eval_correct_by_class[i], g_eval_total_by_class[i],
                g_eval_total_by_class[i] > 0
                    ? (100.0f * g_eval_correct_by_class[i]) / g_eval_total_by_class[i]
                    : 0.0f);
        }
        if (class_len > 0 && len + class_len < (int)sizeof(response)) {
            len += class_len;
        }
    }

    snprintf(response + len, sizeof(response) - len, "]}");

    send_json_response(client_fd, response);
}

void handle_api_indices(int client_fd, const char* query) {
    if (!g_w1 || !g_b1 || !g_w2 || !g_b2) {
        send_json_response(client_fd, "{\"error\":\"model not loaded\"}");
        return;
    }

    ensure_eval_cache();
    if (!g_eval_cache_ready) {
        send_json_response(client_fd, "{\"error\":\"cache not ready\"}");
        return;
    }

    char filter_buf[32] = "all";
    (void)get_query_param(query, "filter", filter_buf, sizeof(filter_buf));

    int mode = 0;
    if (strcmp(filter_buf, "correct") == 0) mode = 1;
    else if (strcmp(filter_buf, "incorrect") == 0) mode = 2;
    else strcpy(filter_buf, "all");

    int limit = TEST_SIZE;
    char limit_buf[32];
    if (get_query_param(query, "limit", limit_buf, sizeof(limit_buf))) {
        int parsed = atoi(limit_buf);
        if (parsed >= 0 && parsed <= TEST_SIZE) limit = parsed;
    }

    char* buffer = (char*)malloc((size_t)TEST_SIZE * 8 + 512);
    if (!buffer) {
        send_json_response(client_fd, "{\"error\":\"out of memory\"}");
        return;
    }

    char* p = buffer;
    size_t remaining = (size_t)TEST_SIZE * 8 + 512;

    int len = snprintf(
        p,
        remaining,
        "{\"filter\":\"%s\",\"total\":%d,\"indices\":[",
        filter_buf,
        mode == 0 ? TEST_SIZE : (mode == 1 ? g_eval_correct_total : (TEST_SIZE - g_eval_correct_total))
    );
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    int returned = 0;
    for (int idx = 0; idx < TEST_SIZE; idx++) {
        int include = 1;
        if (mode == 1) include = g_correct_cache[idx] ? 1 : 0;
        if (mode == 2) include = g_correct_cache[idx] ? 0 : 1;
        if (!include) continue;
        if (returned >= limit) break;

        len = snprintf(p, remaining, "%s%d", returned > 0 ? "," : "", idx);
        if (len > 0 && (size_t)len < remaining) {
            p += len;
            remaining -= (size_t)len;
            returned++;
        }
    }

    len = snprintf(p, remaining, "],\"returned\":%d}", returned);
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    send_json_response(client_fd, buffer);
    free(buffer);
}

static void parse_json_pixels(const char* body, float* pixels, int* count) {
    *count = 0;
    const char* arr_start = strchr(body, '[');
    if (!arr_start) return;

    const char* p = arr_start + 1;
    while (*p && *count < 784) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) p++;
        if (*p == ']' || *p == '\0') break;

        char* end = NULL;
        float val = strtof(p, &end);
        if (end != p) {
            pixels[*count] = val;
            (*count)++;
            p = end;
        } else {
            p++;
        }
    }
}

void handle_api_predict_pixels(int client_fd, const char* body) {
    if (!g_w1 || !g_b1 || !g_w2 || !g_b2) {
        send_json_response(client_fd, "{\"error\":\"model not loaded\"}");
        return;
    }

    float pixels[784] = {0};
    int pixel_count = 0;
    parse_json_pixels(body, pixels, &pixel_count);

    if (pixel_count != 784) {
        send_json_response(client_fd, "{\"error\":\"expected 784 pixels\"}");
        return;
    }

    ForwardTrace* trace = forward_single(pixels, -1, g_w1, g_b1, g_w2, g_b2);

    char* buffer = (char*)malloc(65536);
    if (!buffer) {
        free_forward_trace(trace);
        send_json_response(client_fd, "{\"error\":\"out of memory\"}");
        return;
    }

    char* p = buffer;
    size_t remaining = 65536;

    int len = snprintf(p, remaining,
        "{"
        "\"true_label\":-1,"
        "\"predicted\":%d,"
        "\"correct\":false,"
        "\"confidence\":%.6f,"
        "\"pixels\":",
        trace->predicted, trace->confidence);

    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->pixels, 784);

    len = snprintf(p, remaining, ",\"hidden\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->hidden, trace->hidden_size);

    len = snprintf(p, remaining, ",\"pre_relu\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->pre_relu, trace->hidden_size);

    len = snprintf(p, remaining, ",\"output\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->output, trace->output_size);

    len = snprintf(p, remaining, ",\"pre_softmax\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->pre_softmax, trace->output_size);

    len = snprintf(p, remaining, ",\"matmul1\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->matmul1, trace->hidden_size);

    len = snprintf(p, remaining, ",\"matmul2\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->matmul2, trace->output_size);

    len = snprintf(p, remaining, "}");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    send_json_response(client_fd, buffer);
    free(buffer);
    free_forward_trace(trace);
}

void handle_api_weights(int client_fd, const char* query) {
    if (!g_w1 || !g_b1 || !g_w2 || !g_b2) {
        send_json_response(client_fd, "{\"error\":\"model not loaded\"}");
        return;
    }

    char neuron_str[32] = "0";
    char layer_str[32] = "1";
    (void)get_query_param(query, "neuron", neuron_str, sizeof(neuron_str));
    (void)get_query_param(query, "layer", layer_str, sizeof(layer_str));

    int neuron_idx = atoi(neuron_str);
    int layer = atoi(layer_str);

    char* buffer = (char*)malloc(65536);
    if (!buffer) {
        send_json_response(client_fd, "{\"error\":\"out of memory\"}");
        return;
    }

    char* p = buffer;
    size_t remaining = 65536;
    int len;

    if (layer == 1) {
        int H = g_w1->data->shape[1];
        if (neuron_idx < 0 || neuron_idx >= H) {
            send_json_response(client_fd, "{\"error\":\"neuron out of range\"}");
            free(buffer);
            return;
        }
        len = snprintf(p, remaining, "{\"layer\":1,\"neuron\":%d,\"weights\":[", neuron_idx);
        if (len > 0 && (size_t)len < remaining) { p += len; remaining -= (size_t)len; }
        for (int i = 0; i < 784; i++) {
            if (i > 0) {
                len = snprintf(p, remaining, ",");
                if (len > 0 && (size_t)len < remaining) { p += len; remaining -= (size_t)len; }
            }
            len = snprintf(p, remaining, "%.6f", g_w1->data->values[i * H + neuron_idx]);
            if (len > 0 && (size_t)len < remaining) { p += len; remaining -= (size_t)len; }
        }
        len = snprintf(p, remaining, "],\"bias\":%.6f}", g_b1->data->values[neuron_idx]);
    } else {
        if (neuron_idx < 0 || neuron_idx >= 10) {
            send_json_response(client_fd, "{\"error\":\"neuron out of range\"}");
            free(buffer);
            return;
        }
        int H = g_w1->data->shape[1];
        len = snprintf(p, remaining, "{\"layer\":2,\"neuron\":%d,\"weights\":[", neuron_idx);
        if (len > 0 && (size_t)len < remaining) { p += len; remaining -= (size_t)len; }
        for (int i = 0; i < H; i++) {
            if (i > 0) {
                len = snprintf(p, remaining, ",");
                if (len > 0 && (size_t)len < remaining) { p += len; remaining -= (size_t)len; }
            }
            len = snprintf(p, remaining, "%.6f", g_w2->data->values[i * 10 + neuron_idx]);
            if (len > 0 && (size_t)len < remaining) { p += len; remaining -= (size_t)len; }
        }
        len = snprintf(p, remaining, "],\"bias\":%.6f}", g_b2->data->values[neuron_idx]);
    }

    if (len > 0 && (size_t)len < remaining) { p += len; remaining -= (size_t)len; }

    send_json_response(client_fd, buffer);
    free(buffer);
}
