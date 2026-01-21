#include "tensor.h"
#include "tensor_web.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>

#define PORT 3000
#define BUFFER_SIZE 8192
#define MAX_PATH 256

typedef struct {
    int client_fd;
    char method[16];
    char path[MAX_PATH];
    char query[MAX_PATH];
} HttpRequest;

static int start_server() {
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(1);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(1);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(1);
    }

    if (listen(server_fd, 10) < 0) {
        perror("listen");
        exit(1);
    }

    return server_fd;
}

void send_response(int client_fd, int status_code, const char* status_text,
                          const char* content_type, const char* body) {
    char header[BUFFER_SIZE];
    int body_len = body ? (int)strlen(body) : 0;

    snprintf(header, sizeof(header),
             "HTTP/1.1 %d %s\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %d\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Connection: close\r\n"
             "\r\n",
             status_code, status_text, content_type, body_len);

    send(client_fd, header, strlen(header), 0);
    if (body && body_len > 0) {
        send(client_fd, body, body_len, 0);
    }
}

static void parse_request(const char* buffer, HttpRequest* req) {
    memset(req, 0, sizeof(HttpRequest));

    const char* start = buffer;
    const char* space = strchr(start, ' ');
    if (!space) return;

    size_t method_len = space - start;
    if (method_len >= 16) return;
    memcpy(req->method, start, method_len);
    req->method[method_len] = '\0';

    start = space + 1;
    space = strchr(start, ' ');
    if (!space) return;

    size_t path_len = space - start;
    size_t max_copy = (path_len < (MAX_PATH - 1)) ? path_len : (MAX_PATH - 1);
    memcpy(req->path, start, max_copy);
    req->path[max_copy] = '\0';

    char* query_start = strchr(req->path, '?');
    if (query_start) {
        *query_start = '\0';
        strncpy(req->query, query_start + 1, MAX_PATH - 1);
    }
}

static void url_decode(char* dst, const char* src) {
    char a, b;
    while (*src) {
        if (*src == '%' && (a = src[1]) && (b = src[2]) &&
            isxdigit(a) && isxdigit(b)) {
            if (a >= 'a') a -= 'a' - 'A';
            if (a >= 'A') a -= ('A' - 10);
            else a -= '0';
            if (b >= 'a') b -= 'a' - 'A';
            if (b >= 'A') b -= ('A' - 10);
            else b -= '0';
            *dst++ = 16 * a + b;
            src += 3;
        } else if (*src == '+') {
            *dst++ = ' ';
            src++;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';
}

static void send_file(int client_fd, const char* filepath, const char* content_type) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        char* not_found = "404 Not Found";
        send_response(client_fd, 404, "Not Found", "text/plain", not_found);
        return;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* content = (char*)malloc((size_t)fsize + 1);
    if (!content) {
        fclose(f);
        send_response(client_fd, 500, "Internal Server Error", "text/plain", "Memory error");
        return;
    }

    fread(content, 1, (size_t)fsize, f);
    fclose(f);
    content[fsize] = '\0';

    send_response(client_fd, 200, "OK", content_type, content);
    free(content);
}

int get_query_param(const char* query, const char* param, char* value, size_t value_len) {
    if (!query || !param) return 0;

    const char* p = query;
    while (*p) {
        const char* key_start = p;
        const char* key_end = strchr(p, '=');
        if (!key_end) break;

        const char* val_start = key_end + 1;
        const char* val_end = strchr(val_start, '&');
        if (!val_end) val_end = val_start + strlen(val_start);

        size_t key_len = key_end - key_start;
        size_t param_len = strlen(param);

        if (key_len == param_len && strncmp(key_start, param, param_len) == 0) {
            size_t val_len = (size_t)(val_end - val_start);
            if (val_len < value_len) {
                url_decode(value, val_start);
                return 1;
            }
        }

        p = val_end;
        if (*p == '&') p++;
    }
    return 0;
}

extern void handle_api_architecture(int client_fd);
extern void handle_api_predict(int client_fd, const char* query);
extern void handle_api_eval(int client_fd);

static void handle_request(int client_fd) {
    char buffer[BUFFER_SIZE];
    memset(buffer, 0, sizeof(buffer));

    ssize_t bytes_read = recv(client_fd, buffer, BUFFER_SIZE - 1, 0);
    if (bytes_read <= 0) {
        close(client_fd);
        return;
    }

    HttpRequest req;
    parse_request(buffer, &req);

    if (strcmp(req.method, "GET") != 0) {
        send_response(client_fd, 405, "Method Not Allowed", "text/plain", "Method not allowed");
        close(client_fd);
        return;
    }

    if (strcmp(req.path, "/") == 0 || strcmp(req.path, "/index.html") == 0) {
        send_file(client_fd, "html/index.html", "text/html");
    } else if (strcmp(req.path, "/api/architecture") == 0) {
        handle_api_architecture(client_fd);
    } else if (strcmp(req.path, "/api/predict") == 0) {
        handle_api_predict(client_fd, req.query);
    } else if (strcmp(req.path, "/api/eval") == 0) {
        handle_api_eval(client_fd);
    } else {
        send_response(client_fd, 404, "Not Found", "text/plain", "Not Found");
    }

    close(client_fd);
}

extern void web_init();

int main() {
    web_init();

    printf("Starting server on port %d...\n", PORT);
    printf("Open http://localhost:%d in your browser\n", PORT);

    int server_fd = start_server();

    while (1) {
        struct sockaddr_in address;
        socklen_t addrlen = sizeof(address);

        int client_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        if (client_fd < 0) {
            perror("accept");
            continue;
        }

        handle_request(client_fd);
    }

    close(server_fd);
    return 0;
}
