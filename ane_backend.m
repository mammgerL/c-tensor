#include "ane_backend.h"
#include <Accelerate/Accelerate.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static char g_last_error[256] = "ANE backend not initialized";
static int g_checked = 0;
static int g_private_framework_found = 0;
static int g_opt_in = 0;

static void ane_probe_once(void) {
    if (g_checked) return;
    g_checked = 1;

    const char* env = getenv("ANE_ENABLE_PRIVATE_API");
    g_opt_in = (env && strcmp(env, "1") == 0) ? 1 : 0;
    if (!g_opt_in) {
        snprintf(g_last_error, sizeof(g_last_error),
            "private ANE API disabled (set ANE_ENABLE_PRIVATE_API=1 to opt-in)");
        return;
    }

    void* h = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_LAZY
    );
    if (h) {
        g_private_framework_found = 1;
        dlclose(h);
        snprintf(g_last_error, sizeof(g_last_error),
            "private ANE framework detected; execution path not implemented yet");
    } else {
        snprintf(g_last_error, sizeof(g_last_error),
            "private ANE framework not found");
    }
}

static void cpu_dense_relu_cblas(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int batch,
    int in_dim,
    int out_dim
) {
    for (int i = 0; i < batch; i++) {
        memcpy(out + (size_t)i * out_dim, b, (size_t)out_dim * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        batch, out_dim, in_dim,
        1.0f,
        x, in_dim,
        w, out_dim,
        1.0f,
        out, out_dim);

    int n = batch * out_dim;
    for (int i = 0; i < n; i++) {
        if (out[i] < 0.0f) out[i] = 0.0f;
    }
}

int ane_backend_is_available(void) {
    ane_probe_once();
    return g_opt_in && g_private_framework_found;
}

const char* ane_backend_last_error(void) {
    ane_probe_once();
    return g_last_error;
}

int ane_dense_relu_forward(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int batch,
    int in_dim,
    int out_dim
) {
    ane_probe_once();

    if (!x || !w || !b || !out || batch <= 0 || in_dim <= 0 || out_dim <= 0) {
        snprintf(g_last_error, sizeof(g_last_error), "invalid ane_dense_relu_forward arguments");
        return ANE_BACKEND_ERROR;
    }

    /*
     M1 framework fallback:
     ANE path is intentionally unimplemented in mainline.
     This keeps the integration points stable while preserving correctness.
    */
    cpu_dense_relu_cblas(x, w, b, out, batch, in_dim, out_dim);

    if (g_opt_in && g_private_framework_found) {
        snprintf(g_last_error, sizeof(g_last_error),
            "private ANE framework detected but runtime bridge is not implemented; used CPU fallback");
    } else {
        snprintf(g_last_error, sizeof(g_last_error),
            "ANE unavailable: used CPU fallback");
    }
    return ANE_BACKEND_FALLBACK;
}
