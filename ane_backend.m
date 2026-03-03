#include "ane_backend.h"
#include <Accelerate/Accelerate.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static char g_last_error[256] = "ANE backend not initialized";
static int g_checked = 0;
static int g_private_framework_found = 0;
static int g_opt_in = 0;
static int g_runtime_ready = 0;
static int g_compile_count = 0;
static int g_cache_hit_count = 0;
static int g_fallback_count = 0;

typedef struct {
    int used;
    int batch;
    int in_dim;
    int out_dim;
} AnekernelCacheEntry;

#define ANE_KERNEL_CACHE_SIZE 16
static AnekernelCacheEntry g_kernel_cache[ANE_KERNEL_CACHE_SIZE];

static Class g_ane_client_cls = Nil;
static Class g_ane_desc_cls = Nil;
static Class g_ane_inmem_cls = Nil;
static Class g_ane_req_cls = Nil;
static Class g_ane_io_cls = Nil;

static int ane_find_cache_slot(int batch, int in_dim, int out_dim) {
    for (int i = 0; i < ANE_KERNEL_CACHE_SIZE; i++) {
        if (g_kernel_cache[i].used &&
            g_kernel_cache[i].batch == batch &&
            g_kernel_cache[i].in_dim == in_dim &&
            g_kernel_cache[i].out_dim == out_dim) {
            return i;
        }
    }
    return -1;
}

static int ane_reserve_cache_slot(int batch, int in_dim, int out_dim) {
    for (int i = 0; i < ANE_KERNEL_CACHE_SIZE; i++) {
        if (!g_kernel_cache[i].used) {
            g_kernel_cache[i].used = 1;
            g_kernel_cache[i].batch = batch;
            g_kernel_cache[i].in_dim = in_dim;
            g_kernel_cache[i].out_dim = out_dim;
            return i;
        }
    }
    return -1;
}

static void ane_try_bind_runtime(void) {
    g_ane_client_cls = objc_getClass("_ANEClient");
    g_ane_desc_cls = objc_getClass("_ANEInMemoryModelDescriptor");
    g_ane_inmem_cls = objc_getClass("_ANEInMemoryModel");
    g_ane_req_cls = objc_getClass("_ANERequest");
    g_ane_io_cls = objc_getClass("_ANEIOSurfaceObject");

    if (g_ane_client_cls && g_ane_desc_cls && g_ane_inmem_cls && g_ane_req_cls && g_ane_io_cls) {
        g_runtime_ready = 1;
    } else {
        g_runtime_ready = 0;
    }
}

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
        ane_try_bind_runtime();
        dlclose(h);
        if (g_runtime_ready) {
            snprintf(g_last_error, sizeof(g_last_error),
                "private ANE framework detected and runtime classes resolved; execution bridge not implemented");
        } else {
            snprintf(g_last_error, sizeof(g_last_error),
                "private ANE framework detected but required classes are missing");
        }
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
    return g_opt_in && g_private_framework_found && g_runtime_ready;
}

const char* ane_backend_last_error(void) {
    ane_probe_once();
    return g_last_error;
}

void ane_backend_get_stats(int* compile_count, int* cache_hit_count, int* fallback_count) {
    if (compile_count) *compile_count = g_compile_count;
    if (cache_hit_count) *cache_hit_count = g_cache_hit_count;
    if (fallback_count) *fallback_count = g_fallback_count;
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
    if (ane_find_cache_slot(batch, in_dim, out_dim) >= 0) {
        g_cache_hit_count++;
    } else if (ane_reserve_cache_slot(batch, in_dim, out_dim) >= 0) {
        g_compile_count++;
    }

    cpu_dense_relu_cblas(x, w, b, out, batch, in_dim, out_dim);
    g_fallback_count++;

    if (g_opt_in && g_private_framework_found && g_runtime_ready) {
        snprintf(g_last_error, sizeof(g_last_error),
            "private ANE framework detected but runtime bridge is not implemented; used CPU fallback");
    } else if (g_opt_in && g_private_framework_found && !g_runtime_ready) {
        snprintf(g_last_error, sizeof(g_last_error),
            "private ANE framework found but required classes unavailable; used CPU fallback");
    } else {
        snprintf(g_last_error, sizeof(g_last_error),
            "ANE unavailable: used CPU fallback");
    }
    return ANE_BACKEND_FALLBACK;
}
