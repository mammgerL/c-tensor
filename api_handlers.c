#include "tensor.h"
#include "tensor_web.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

void send_response(int client_fd, int status_code, const char* status_text,
                        const char* content_type, const char* body);
int get_query_param(const char* query, const char* param, char* value, size_t value_len);

static Tensor* g_w1 = NULL, * g_b1 = NULL, * g_w2 = NULL, * g_b2 = NULL;
static Tensor* g_test_x = NULL, * g_test_y = NULL;
static const int TEST_SIZE = 10000;

extern void send_response(int client_fd, int status_code, const char* status_text,
                        const char* content_type, const char* body);
extern int get_query_param(const char* query, const char* param, char* value, size_t value_len);

void web_init() {
    load_model("mnist_mlp.bin", &g_w1, &g_b1, &g_w2, &g_b2);

    g_test_x = create_zero_tensor((int[]) { TEST_SIZE, 784 }, 2);
    g_test_y = create_zero_tensor((int[]) { TEST_SIZE, 10 }, 2);

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

    char* buffer = (char*)malloc(131072);
    if (!buffer) {
        free_forward_trace(trace);
        send_json_response(client_fd, "{\"error\":\"out of memory\"}");
        return;
    }

    char* p = buffer;
    size_t remaining = 131072;
    int H = g_w1->data->shape[1];

    int len = snprintf(p, remaining,
        "{"
        "\"index\":%d,"
        "\"true_label\":%d,"
        "\"predicted\":%d,"
        "\"correct\":%s,"
        "\"confidence\":%.6f,"
        "\"pixels\":",
        index, trace->true_label, trace->predicted,
        trace->predicted == trace->true_label ? "true" : "false",
        trace->confidence);

    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->pixels, 784);

    len = snprintf(p, remaining,
        ",\"hidden\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->hidden, trace->hidden_size);

    len = snprintf(p, remaining,
        ",\"output\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->output, trace->output_size);

    len = snprintf(p, remaining,
        ",\"computation\":{"
        "\"steps\":[");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    len = snprintf(p, remaining,
        "{"
        "\"name\":\"Layer 1: Input -> Hidden\","
        "\"operation\":\"matmul + bias + relu\","
        "\"input\":{\"shape\":[1,784],\"type\":\"image_pixels\"},"
        "\"weight\":{\"shape\":[784,%d],\"sample\":[", H);
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    for (int i = 0; i < 5 && i < H; i++) {
        append_float(&p, &remaining, g_w1->data->values[i]);
        if (i < 4 && i < H - 1) {
            len = snprintf(p, remaining, ",");
            if (len > 0 && (size_t)len < remaining) {
                p += len;
                remaining -= (size_t)len;
            }
        }
    }

    len = snprintf(p, remaining,
        "]},"
        "\"bias\":{\"shape\":[%d],\"sample\":[", H);
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    for (int i = 0; i < 5 && i < H; i++) {
        append_float(&p, &remaining, g_b1->data->values[i]);
        if (i < 4 && i < H - 1) {
            len = snprintf(p, remaining, ",");
            if (len > 0 && (size_t)len < remaining) {
                p += len;
                remaining -= (size_t)len;
            }
        }
    }

    len = snprintf(p, remaining,
        "]},"
        "\"output\":{\"shape\":[1,%d],\"activations\":", H);
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->hidden, H);

    len = snprintf(p, remaining,
        "}},{"
        "\"name\":\"Layer 2: Hidden -> Output\","
        "\"operation\":\"matmul + bias + logsoftmax\","
        "\"input\":{\"shape\":[1,%d],\"type\":\"relu_activations\"},"
        "\"weight\":{\"shape\":[%d,10],\"sample\":[", H, H);
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    for (int i = 0; i < 5 && i < H; i++) {
        for (int j = 0; j < 2; j++) {
            append_float(&p, &remaining, g_w2->data->values[i * 10 + j]);
            if (j == 0) {
                len = snprintf(p, remaining, ",");
                if (len > 0 && (size_t)len < remaining) {
                    p += len;
                    remaining -= (size_t)len;
                }
            }
        }
        if (i < 4 && i < H - 1) {
            len = snprintf(p, remaining, ",");
            if (len > 0 && (size_t)len < remaining) {
                p += len;
                remaining -= (size_t)len;
            }
        }
    }

    len = snprintf(p, remaining,
        "]},"
        "\"bias\":{\"shape\":[10],\"sample\":[");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    for (int i = 0; i < 5 && i < 10; i++) {
        append_float(&p, &remaining, g_b2->data->values[i]);
        if (i < 4) {
            len = snprintf(p, remaining, ",");
            if (len > 0 && (size_t)len < remaining) {
                p += len;
                remaining -= (size_t)len;
            }
        }
    }

    len = snprintf(p, remaining,
        "]},"
        "\"output\":{\"shape\":[1,10],\"activations\":");
    if (len > 0 && (size_t)len < remaining) {
        p += len;
        remaining -= (size_t)len;
    }

    append_float_array(&p, &remaining, trace->output, 10);

    len = snprintf(p, remaining,
        "}}]}}");
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

    int correct = 0;
    int total = 0;
    int* correct_by_class = (int*)calloc(10, sizeof(int));
    int* total_by_class = (int*)calloc(10, sizeof(int));

    const int BATCH_SIZE = 256;

    for (int start = 0; start < TEST_SIZE; start += BATCH_SIZE) {
        int curB = (start + BATCH_SIZE <= TEST_SIZE) ? BATCH_SIZE : (TEST_SIZE - start);

        for (int i = 0; i < curB; i++) {
            int idx = start + i;
            const float* x_data = g_test_x->data->values + (size_t)idx * 784;

            int true_label = -1;
            for (int c = 0; c < 10; c++) {
                if (g_test_y->data->values[idx * 10 + c] < 0) {
                    true_label = c;
                    break;
                }
            }

            Tensor* x = create_zero_tensor((int[]) { 1, 784 }, 2);
            memcpy(x->data->values, x_data, 784 * sizeof(float));

            Tensor* h1 = matmul(x, g_w1);
            Tensor* h1b = add_bias(h1, g_b1);
            Tensor* r1 = relu(h1b);
            Tensor* h2 = matmul(r1, g_w2);
            Tensor* h2b = add_bias(h2, g_b2);
            Tensor* lout = logsoftmax(h2b);

            int predicted = argmax(lout->data->values, 10);

            total++;
            total_by_class[true_label]++;

            if (predicted == true_label) {
                correct++;
                correct_by_class[true_label]++;
            }

            free_tensor(x);
            free_tensor(h1);
            free_tensor(h1b);
            free_tensor(r1);
            free_tensor(h2);
            free_tensor(h2b);
            free_tensor(lout);
        }
    }

    char response[4096];
    float accuracy = 100.0f * (float)correct / (float)total;

    int len = snprintf(response, sizeof(response),
        "{"
        "\"total\":%d,"
        "\"correct\":%d,"
        "\"accuracy\":%.2f,"
        "\"incorrect\":%d,"
        "\"by_class\":[",
        total, correct, accuracy, total - correct);

    for (int i = 0; i < 10; i++) {
        int class_len;
        if (i == 0) {
            class_len = snprintf(response + len, sizeof(response) - len,
                "{\"digit\":%d,\"correct\":%d,\"total\":%d,\"accuracy\":%.2f}",
                i, correct_by_class[i], total_by_class[i],
                total_by_class[i] > 0 ? (100.0f * correct_by_class[i]) / total_by_class[i] : 0.0f);
        } else {
            class_len = snprintf(response + len, sizeof(response) - len,
                ",{\"digit\":%d,\"correct\":%d,\"total\":%d,\"accuracy\":%.2f}",
                i, correct_by_class[i], total_by_class[i],
                total_by_class[i] > 0 ? (100.0f * correct_by_class[i]) / total_by_class[i] : 0.0f);
        }
        if (class_len > 0 && len + class_len < (int)sizeof(response)) {
            len += class_len;
        }
    }

    snprintf(response + len, sizeof(response) - len, "]}");

    send_json_response(client_fd, response);

    free(correct_by_class);
    free(total_by_class);
}
