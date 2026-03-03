#include "ane_backend.h"
#include <Accelerate/Accelerate.h>
#include <Foundation/Foundation.h>
#include <IOSurface/IOSurface.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_last_error[256] = "ANE backend not initialized";
static int g_checked = 0;
static int g_private_framework_found = 0;
static int g_opt_in = 0;
static int g_runtime_ready = 0;
static int g_compile_count = 0;
static int g_cache_hit_count = 0;
static int g_fallback_count = 0;

static Class g_ane_client_cls = Nil;
static Class g_ane_desc_cls = Nil;
static Class g_ane_inmem_cls = Nil;
static Class g_ane_req_cls = Nil;
static Class g_ane_io_cls = Nil;

typedef struct {
    int used;
    int batch;
    int in_dim;
    int out_dim;
    uint64_t weight_hash;
    id model;
    id request;
    IOSurfaceRef in_surface;
    IOSurfaceRef out_surface;
    NSString* tmp_dir;
    float* in_cf;
    float* out_cf;
} AnekernelCacheEntry;

#define ANE_KERNEL_CACHE_SIZE 4
static AnekernelCacheEntry g_kernel_cache[ANE_KERNEL_CACHE_SIZE];
static int g_evict_cursor = 0;

static void set_last_error(const char* msg) {
    snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
}

static void set_last_errorf(const char* prefix, const char* detail) {
    snprintf(g_last_error, sizeof(g_last_error), "%s: %s", prefix, detail ? detail : "(null)");
}

static IOSurfaceRef ane_create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

static NSString* ane_dense_mil(int batch, int in_dim, int out_dim) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=string(\"conv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, batch,
        in_dim, batch,
        out_dim, in_dim, out_dim, in_dim,
        out_dim, batch,
        out_dim, batch];
}

static uint64_t hash_weights(const float* w, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        uint32_t u = 0;
        memcpy(&u, &w[i], sizeof(uint32_t));
        h ^= (uint64_t)u;
        h *= 1099511628211ULL;
    }
    return h;
}

static NSData* ane_build_weight_blob_from_io(const float* w_io, int in_dim, int out_dim) {
    size_t wsize = (size_t)in_dim * out_dim * 2;
    size_t total = 128 + wsize;
    uint8_t* buf = (uint8_t*)calloc(total, 1);
    if (!buf) return nil;

    buf[0] = 0x01;
    buf[4] = 0x02;
    uint8_t* chunk = buf + 64;
    chunk[0] = 0xEF;
    chunk[1] = 0xBE;
    chunk[2] = 0xAD;
    chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128;

    _Float16* fp16 = (_Float16*)(buf + 128);
    for (int o = 0; o < out_dim; o++) {
        for (int i = 0; i < in_dim; i++) {
            fp16[(size_t)o * in_dim + i] = (_Float16)w_io[(size_t)i * out_dim + o];
        }
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static void ane_free_kernel(AnekernelCacheEntry* k) {
    if (!k || !k->used) return;

    NSError* e = nil;
    if (k->model) {
        ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
            k->model, @selector(unloadWithQoS:error:), 21, &e
        );
    }

    if (k->in_surface) CFRelease(k->in_surface);
    if (k->out_surface) CFRelease(k->out_surface);

    if (k->tmp_dir) {
        [[NSFileManager defaultManager] removeItemAtPath:k->tmp_dir error:nil];
    }

    free(k->in_cf);
    free(k->out_cf);
    memset(k, 0, sizeof(*k));
}

static int ane_find_kernel_slot(int batch, int in_dim, int out_dim, uint64_t weight_hash) {
    for (int i = 0; i < ANE_KERNEL_CACHE_SIZE; i++) {
        AnekernelCacheEntry* k = &g_kernel_cache[i];
        if (k->used &&
            k->batch == batch &&
            k->in_dim == in_dim &&
            k->out_dim == out_dim &&
            k->weight_hash == weight_hash) {
            return i;
        }
    }
    return -1;
}

static int ane_prepare_runtime_entry(
    AnekernelCacheEntry* k,
    int batch,
    int in_dim,
    int out_dim,
    uint64_t weight_hash,
    const float* w_io
) {
    NSString* mil = ane_dense_mil(batch, in_dim, out_dim);
    NSData* mil_data = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSData* weight_data = ane_build_weight_blob_from_io(w_io, in_dim, out_dim);
    if (!weight_data) {
        set_last_error("failed to allocate weight blob");
        return 0;
    }
    NSDictionary* wdict = @{
        @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weight_data}
    };

    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        g_ane_desc_cls, @selector(modelWithMILText:weights:optionsPlist:), mil_data, wdict, nil
    );
    if (!desc) {
        set_last_error("ANE descriptor creation failed");
        return 0;
    }

    id model = ((id(*)(Class, SEL, id))objc_msgSend)(
        g_ane_inmem_cls, @selector(inMemoryModelWithDescriptor:), desc
    );
    if (!model) {
        set_last_error("ANE in-memory model creation failed");
        return 0;
    }

    id hx = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString* td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager* fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil_data writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [weight_data writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError* e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e
    );
    if (!ok) {
        set_last_errorf("ANE compile failed", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    e = nil;
    ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e
    );
    if (!ok) {
        set_last_errorf("ANE load failed", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    size_t in_bytes = (size_t)batch * in_dim * sizeof(float);
    size_t out_bytes = (size_t)batch * out_dim * sizeof(float);

    IOSurfaceRef in_surface = ane_create_surface(in_bytes);
    IOSurfaceRef out_surface = ane_create_surface(out_bytes);
    if (!in_surface || !out_surface) {
        set_last_error("failed to create IOSurface");
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    id w_in = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        g_ane_io_cls, @selector(objectWithIOSurface:), in_surface
    );
    id w_out = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        g_ane_io_cls, @selector(objectWithIOSurface:), out_surface
    );

    id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        g_ane_req_cls,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[w_in], @[@0], @[w_out], @[@0], nil, nil, @0
    );
    if (!request) {
        set_last_error("failed to create ANE request");
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    float* in_cf = (float*)malloc(in_bytes);
    float* out_cf = (float*)malloc(out_bytes);
    if (!in_cf || !out_cf) {
        free(in_cf);
        free(out_cf);
        set_last_error("failed to allocate ANE reorder buffers");
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    k->used = 1;
    k->batch = batch;
    k->in_dim = in_dim;
    k->out_dim = out_dim;
    k->weight_hash = weight_hash;
    k->model = model;
    k->request = request;
    k->in_surface = in_surface;
    k->out_surface = out_surface;
    k->tmp_dir = td;
    k->in_cf = in_cf;
    k->out_cf = out_cf;
    return 1;
}

static AnekernelCacheEntry* ane_get_kernel(
    int batch,
    int in_dim,
    int out_dim,
    const float* w_io
) {
    uint64_t wh = hash_weights(w_io, in_dim * out_dim);
    int hit = ane_find_kernel_slot(batch, in_dim, out_dim, wh);
    if (hit >= 0) {
        g_cache_hit_count++;
        return &g_kernel_cache[hit];
    }

    int slot = -1;
    for (int i = 0; i < ANE_KERNEL_CACHE_SIZE; i++) {
        if (!g_kernel_cache[i].used) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        slot = g_evict_cursor % ANE_KERNEL_CACHE_SIZE;
        g_evict_cursor++;
        ane_free_kernel(&g_kernel_cache[slot]);
    }

    if (!ane_prepare_runtime_entry(&g_kernel_cache[slot], batch, in_dim, out_dim, wh, w_io)) {
        ane_free_kernel(&g_kernel_cache[slot]);
        return NULL;
    }

    g_compile_count++;
    return &g_kernel_cache[slot];
}

static int ane_eval_dense(
    AnekernelCacheEntry* k,
    const float* x_rowmajor,
    const float* b,
    float* out_rowmajor
) {
    int batch = k->batch;
    int in_dim = k->in_dim;
    int out_dim = k->out_dim;

    for (int s = 0; s < batch; s++) {
        const float* xr = x_rowmajor + (size_t)s * in_dim;
        for (int c = 0; c < in_dim; c++) {
            k->in_cf[(size_t)c * batch + s] = xr[c];
        }
    }

    size_t in_bytes = (size_t)batch * in_dim * sizeof(float);
    size_t out_bytes = (size_t)batch * out_dim * sizeof(float);

    IOSurfaceLock(k->in_surface, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->in_surface), k->in_cf, in_bytes);
    IOSurfaceUnlock(k->in_surface, 0, NULL);

    NSError* e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e
    );
    if (!ok) {
        set_last_errorf("ANE evaluate failed", [[e description] UTF8String]);
        return 0;
    }

    IOSurfaceLock(k->out_surface, kIOSurfaceLockReadOnly, NULL);
    memcpy(k->out_cf, IOSurfaceGetBaseAddress(k->out_surface), out_bytes);
    IOSurfaceUnlock(k->out_surface, kIOSurfaceLockReadOnly, NULL);

    for (int s = 0; s < batch; s++) {
        float* o = out_rowmajor + (size_t)s * out_dim;
        for (int c = 0; c < out_dim; c++) {
            float v = k->out_cf[(size_t)c * batch + s] + b[c];
            o[c] = (v > 0.0f) ? v : 0.0f;
        }
    }

    return 1;
}

static void ane_try_bind_runtime(void) {
    g_ane_client_cls = objc_getClass("_ANEClient");
    g_ane_desc_cls = objc_getClass("_ANEInMemoryModelDescriptor");
    g_ane_inmem_cls = objc_getClass("_ANEInMemoryModel");
    g_ane_req_cls = objc_getClass("_ANERequest");
    g_ane_io_cls = objc_getClass("_ANEIOSurfaceObject");
    g_runtime_ready = (g_ane_client_cls && g_ane_desc_cls && g_ane_inmem_cls && g_ane_req_cls && g_ane_io_cls) ? 1 : 0;
}

static void ane_probe_once(void) {
    if (g_checked) return;
    g_checked = 1;

    const char* env = getenv("ANE_ENABLE_PRIVATE_API");
    g_opt_in = (env && strcmp(env, "1") == 0) ? 1 : 0;
    if (!g_opt_in) {
        set_last_error("private ANE API disabled (set ANE_ENABLE_PRIVATE_API=1 to opt-in)");
        return;
    }

    void* h = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    );
    if (!h) {
        set_last_error("private ANE framework not found");
        return;
    }

    g_private_framework_found = 1;
    ane_try_bind_runtime();
    dlclose(h);

    if (g_runtime_ready) {
        set_last_error("private ANE runtime ready");
    } else {
        set_last_error("private ANE framework found but required classes are missing");
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
        set_last_error("invalid ane_dense_relu_forward arguments");
        return ANE_BACKEND_ERROR;
    }

    if (g_opt_in && g_private_framework_found && g_runtime_ready) {
        AnekernelCacheEntry* k = ane_get_kernel(batch, in_dim, out_dim, w);
        if (k && ane_eval_dense(k, x, b, out)) {
            set_last_error("ANE eval OK (dense on ANE, bias/relu on CPU)");
            return ANE_BACKEND_OK;
        }
    }

    cpu_dense_relu_cblas(x, w, b, out, batch, in_dim, out_dim);
    g_fallback_count++;

    if (g_opt_in && g_private_framework_found && g_runtime_ready) {
        if (strstr(g_last_error, "ANE compile failed") == NULL &&
            strstr(g_last_error, "ANE load failed") == NULL &&
            strstr(g_last_error, "ANE evaluate failed") == NULL &&
            strstr(g_last_error, "descriptor") == NULL &&
            strstr(g_last_error, "request") == NULL &&
            strstr(g_last_error, "IOSurface") == NULL) {
            set_last_error("ANE path failed, used CPU fallback");
        }
    } else if (g_opt_in && g_private_framework_found && !g_runtime_ready) {
        set_last_error("private ANE framework found but required classes unavailable; used CPU fallback");
    } else {
        set_last_error("ANE unavailable: used CPU fallback");
    }

    return ANE_BACKEND_FALLBACK;
}
