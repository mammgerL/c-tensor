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
static int g_dynamic_mode = 0;
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
} AnekernelCacheEntry;

#define ANE_KERNEL_CACHE_SIZE 4
static AnekernelCacheEntry g_kernel_cache[ANE_KERNEL_CACHE_SIZE];
static int g_evict_cursor = 0;

typedef struct {
    int used;
    int batch;
    int in_dim;
    int hidden_dim;
    int out_dim;
    id model;
    id request;
    IOSurfaceRef in_surface;
    IOSurfaceRef r1_surface;
    IOSurfaceRef logits_surface;
    NSString* tmp_dir;
} AneMlp2CacheEntry;

#define ANE_MLP2_CACHE_SIZE 2
static AneMlp2CacheEntry g_mlp2_cache[ANE_MLP2_CACHE_SIZE];
static int g_mlp2_evict_cursor = 0;

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

static NSString* ane_dense_mil_dynamic(int batch, int in_dim, int out_dim) {
    NSMutableString* m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    int sp_total = batch + out_dim;
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", in_dim, sp_total];
    [m appendString:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n", in_dim, sp_total];
    [m appendString:@"        tensor<int32, [4]> ba = const()[name = string(\"ba\"), val = tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", in_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n", in_dim, batch];
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,0,0,%d])];\n", batch];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", in_dim, out_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wt\")];\n", in_dim, out_dim];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", in_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", in_dim, batch];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", batch, in_dim];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name = string(\"rw\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", in_dim, out_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", in_dim, out_dim];
    [m appendString:@"        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"mm\")];\n", batch, out_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", out_dim, batch];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name = string(\"ro\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", out_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n", out_dim, batch];
    [m appendString:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> y = cast(dtype = to32, x = yr)[name=string(\"cout\")];\n", out_dim, batch];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

static NSString* ane_mlp2_mil_dynamic(int batch, int in_dim, int hidden_dim, int out_dim) {
    int len_x = in_dim * batch;
    int len_w1 = in_dim * hidden_dim;
    int len_b1 = hidden_dim;
    int len_w2 = hidden_dim * out_dim;
    int len_b2 = out_dim;

    int off_x = 0;
    int off_w1 = off_x + len_x;
    int off_b1 = off_w1 + len_w1;
    int off_w2 = off_b1 + len_b1;
    int off_b2 = off_w2 + len_w2;
    int total = off_b2 + len_b2;

    NSMutableString* m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, 1, 1, %d]> inp) {\n", total];
    [m appendString:@"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"];
    [m appendString:@"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> xh = cast(dtype = to16, x = inp)[name = string(\"cin\")];\n", total];

    [m appendFormat:@"        tensor<int32, [4]> bx = const()[name = string(\"bx\"), val = tensor<int32, [4]>([0,0,0,%d])];\n", off_x];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name = string(\"sx\"), val = tensor<int32, [4]>([1,1,1,%d])];\n", len_x];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> xflat = slice_by_size(x=xh,begin=bx,size=sx)[name=string(\"xflat\")];\n", len_x];

    [m appendFormat:@"        tensor<int32, [4]> bw1 = const()[name = string(\"bw1\"), val = tensor<int32, [4]>([0,0,0,%d])];\n", off_w1];
    [m appendFormat:@"        tensor<int32, [4]> sw1 = const()[name = string(\"sw1\"), val = tensor<int32, [4]>([1,1,1,%d])];\n", len_w1];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> w1flat = slice_by_size(x=xh,begin=bw1,size=sw1)[name=string(\"w1flat\")];\n", len_w1];

    [m appendFormat:@"        tensor<int32, [4]> bb1 = const()[name = string(\"bb1\"), val = tensor<int32, [4]>([0,0,0,%d])];\n", off_b1];
    [m appendFormat:@"        tensor<int32, [4]> sb1 = const()[name = string(\"sb1\"), val = tensor<int32, [4]>([1,1,1,%d])];\n", len_b1];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> b1flat = slice_by_size(x=xh,begin=bb1,size=sb1)[name=string(\"b1flat\")];\n", len_b1];

    [m appendFormat:@"        tensor<int32, [4]> bw2 = const()[name = string(\"bw2\"), val = tensor<int32, [4]>([0,0,0,%d])];\n", off_w2];
    [m appendFormat:@"        tensor<int32, [4]> sw2 = const()[name = string(\"sw2\"), val = tensor<int32, [4]>([1,1,1,%d])];\n", len_w2];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> w2flat = slice_by_size(x=xh,begin=bw2,size=sw2)[name=string(\"w2flat\")];\n", len_w2];

    [m appendFormat:@"        tensor<int32, [4]> bb2 = const()[name = string(\"bb2\"), val = tensor<int32, [4]>([0,0,0,%d])];\n", off_b2];
    [m appendFormat:@"        tensor<int32, [4]> sb2 = const()[name = string(\"sb2\"), val = tensor<int32, [4]>([1,1,1,%d])];\n", len_b2];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> b2flat = slice_by_size(x=xh,begin=bb2,size=sb2)[name=string(\"b2flat\")];\n", len_b2];

    [m appendFormat:@"        tensor<int32, [4]> rx = const()[name = string(\"rx\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", in_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xcf = reshape(shape=rx,x=xflat)[name=string(\"xcf\")];\n", in_dim, batch];
    [m appendFormat:@"        tensor<int32, [4]> rw1 = const()[name = string(\"rw1\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", in_dim, hidden_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> w1cf = reshape(shape=rw1,x=w1flat)[name=string(\"w1cf\")];\n", in_dim, hidden_dim];
    [m appendFormat:@"        tensor<int32, [4]> rb1 = const()[name = string(\"rb1\"), val = tensor<int32, [4]>([1,%d,1,1])];\n", hidden_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> b1cf = reshape(shape=rb1,x=b1flat)[name=string(\"b1cf\")];\n", hidden_dim];
    [m appendFormat:@"        tensor<int32, [4]> rw2 = const()[name = string(\"rw2\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", hidden_dim, out_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> w2cf = reshape(shape=rw2,x=w2flat)[name=string(\"w2cf\")];\n", hidden_dim, out_dim];
    [m appendFormat:@"        tensor<int32, [4]> rb2 = const()[name = string(\"rb2\"), val = tensor<int32, [4]>([1,%d,1,1])];\n", out_dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> b2cf = reshape(shape=rb2,x=b2flat)[name=string(\"b2cf\")];\n", out_dim];

    [m appendFormat:@"        tensor<int32, [4]> rx2 = const()[name = string(\"rx2\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", in_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x2 = reshape(shape=rx2,x=xcf)[name=string(\"x2\")];\n", in_dim, batch];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> x3 = transpose(perm=pm,x=x2)[name=string(\"x3\")];\n", batch, in_dim];
    [m appendFormat:@"        tensor<int32, [4]> rw12 = const()[name = string(\"rw12\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", in_dim, hidden_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W1 = reshape(shape=rw12,x=w1cf)[name=string(\"W1\")];\n", in_dim, hidden_dim];
    [m appendString:@"        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=x3,y=W1)[name=string(\"h1m\")];\n", batch, hidden_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];\n", hidden_dim, batch];
    [m appendFormat:@"        tensor<int32, [4]> rh1 = const()[name = string(\"rh1\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", hidden_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = reshape(shape=rh1,x=h1t)[name=string(\"h1\")];\n", hidden_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1b = add(x=h1,y=b1cf)[name=string(\"h1b\")];\n", hidden_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> r1 = relu(x=h1b)[name=string(\"r1\")];\n", hidden_dim, batch];

    [m appendFormat:@"        tensor<int32, [4]> rr12 = const()[name = string(\"rr12\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", hidden_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> r12 = reshape(shape=rr12,x=r1)[name=string(\"r12\")];\n", hidden_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> r13 = transpose(perm=pm,x=r12)[name=string(\"r13\")];\n", batch, hidden_dim];
    [m appendFormat:@"        tensor<int32, [4]> rw22 = const()[name = string(\"rw22\"), val = tensor<int32, [4]>([1,1,%d,%d])];\n", hidden_dim, out_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W2 = reshape(shape=rw22,x=w2cf)[name=string(\"W2\")];\n", hidden_dim, out_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h2m = matmul(transpose_x=bF,transpose_y=bF,x=r13,y=W2)[name=string(\"h2m\")];\n", batch, out_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h2t = transpose(perm=pm,x=h2m)[name=string(\"h2t\")];\n", out_dim, batch];
    [m appendFormat:@"        tensor<int32, [4]> rh2 = const()[name = string(\"rh2\"), val = tensor<int32, [4]>([1,%d,1,%d])];\n", out_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h2 = reshape(shape=rh2,x=h2t)[name=string(\"h2\")];\n", out_dim, batch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> logits16 = add(x=h2,y=b2cf)[name=string(\"logits16\")];\n", out_dim, batch];

    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> r1o = cast(dtype=to32,x=r1)[name=string(\"r1o\")];\n", hidden_dim, batch];
    [m appendFormat:@"        tensor<fp32, [1,%d,1,%d]> logo = cast(dtype=to32,x=logits16)[name=string(\"logo\")];\n", out_dim, batch];
    [m appendString:@"    } -> (r1o,logo);\n}\n"];
    return m;
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

    k->used = 0;
    k->batch = 0;
    k->in_dim = 0;
    k->out_dim = 0;
    k->weight_hash = 0ULL;
    k->model = nil;
    k->request = nil;
    k->in_surface = NULL;
    k->out_surface = NULL;
    k->tmp_dir = nil;
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
    NSString* mil = g_dynamic_mode ? ane_dense_mil_dynamic(batch, in_dim, out_dim)
                                   : ane_dense_mil(batch, in_dim, out_dim);
    NSData* mil_data = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary* wdict = nil;
    NSData* weight_data = nil;
    if (!g_dynamic_mode) {
        weight_data = ane_build_weight_blob_from_io(w_io, in_dim, out_dim);
        if (!weight_data) {
            set_last_error("failed to allocate weight blob");
            return 0;
        }
        wdict = @{
            @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weight_data}
        };
    } else {
        // Dynamic matmul path does not use baked constants, but descriptor creation expects a map.
        wdict = @{};
    }

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
    if (!g_dynamic_mode) {
        [weight_data writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    }

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

    size_t in_bytes = g_dynamic_mode
        ? (size_t)in_dim * (batch + out_dim) * sizeof(float)
        : (size_t)batch * in_dim * sizeof(float);
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
    return 1;
}

static AnekernelCacheEntry* ane_get_kernel(
    int batch,
    int in_dim,
    int out_dim,
    const float* w_io
) {
    uint64_t wh = g_dynamic_mode ? 0ULL : hash_weights(w_io, in_dim * out_dim);
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
    const float* w_rowmajor,
    const float* b,
    float* out_rowmajor,
    int apply_relu
) {
    int batch = k->batch;
    int in_dim = k->in_dim;
    int out_dim = k->out_dim;

    IOSurfaceLock(k->in_surface, 0, NULL);
    float* in_base = (float*)IOSurfaceGetBaseAddress(k->in_surface);
    if (g_dynamic_mode) {
        int sp = batch + out_dim;
        for (int c = 0; c < in_dim; c++) {
            for (int s = 0; s < batch; s++) {
                in_base[(size_t)c * sp + s] = x_rowmajor[(size_t)s * in_dim + c];
            }
            memcpy(
                in_base + (size_t)c * sp + batch,
                w_rowmajor + (size_t)c * out_dim,
                (size_t)out_dim * sizeof(float)
            );
        }
    } else {
        for (int s = 0; s < batch; s++) {
            const float* xr = x_rowmajor + (size_t)s * in_dim;
            for (int c = 0; c < in_dim; c++) {
                in_base[(size_t)c * batch + s] = xr[c];
            }
        }
    }
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
    float* out_cf = (float*)IOSurfaceGetBaseAddress(k->out_surface);
    for (int s = 0; s < batch; s++) {
        float* o = out_rowmajor + (size_t)s * out_dim;
        for (int c = 0; c < out_dim; c++) {
            float v = out_cf[(size_t)c * batch + s] + b[c];
            o[c] = apply_relu ? ((v > 0.0f) ? v : 0.0f) : v;
        }
    }
    IOSurfaceUnlock(k->out_surface, kIOSurfaceLockReadOnly, NULL);

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
    const char* dyn = getenv("ANE_DYNAMIC_WEIGHTS");
    g_dynamic_mode = (dyn && strcmp(dyn, "1") == 0) ? 1 : 0;
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
        set_last_error(g_dynamic_mode
            ? "private ANE runtime ready (dynamic weights mode)"
            : "private ANE runtime ready");
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

static void cpu_dense_bias_cblas(
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
}

static void ane_free_mlp2_kernel(AneMlp2CacheEntry* k) {
    if (!k || !k->used) return;

    NSError* e = nil;
    if (k->model) {
        ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
            k->model, @selector(unloadWithQoS:error:), 21, &e
        );
    }

    if (k->in_surface) CFRelease(k->in_surface);
    if (k->r1_surface) CFRelease(k->r1_surface);
    if (k->logits_surface) CFRelease(k->logits_surface);

    if (k->tmp_dir) {
        [[NSFileManager defaultManager] removeItemAtPath:k->tmp_dir error:nil];
    }

    k->used = 0;
    k->batch = 0;
    k->in_dim = 0;
    k->hidden_dim = 0;
    k->out_dim = 0;
    k->model = nil;
    k->request = nil;
    k->in_surface = NULL;
    k->r1_surface = NULL;
    k->logits_surface = NULL;
    k->tmp_dir = nil;
}

static int ane_find_mlp2_slot(int batch, int in_dim, int hidden_dim, int out_dim) {
    for (int i = 0; i < ANE_MLP2_CACHE_SIZE; i++) {
        AneMlp2CacheEntry* k = &g_mlp2_cache[i];
        if (k->used &&
            k->batch == batch &&
            k->in_dim == in_dim &&
            k->hidden_dim == hidden_dim &&
            k->out_dim == out_dim) {
            return i;
        }
    }
    return -1;
}

static int ane_prepare_mlp2_entry(
    AneMlp2CacheEntry* k,
    int batch,
    int in_dim,
    int hidden_dim,
    int out_dim
) {
    NSString* mil = ane_mlp2_mil_dynamic(batch, in_dim, hidden_dim, out_dim);
    NSData* mil_data = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary* wdict = @{};

    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        g_ane_desc_cls, @selector(modelWithMILText:weights:optionsPlist:), mil_data, wdict, nil
    );
    if (!desc) {
        set_last_error("ANE fused2 descriptor creation failed");
        return 0;
    }

    id model = ((id(*)(Class, SEL, id))objc_msgSend)(
        g_ane_inmem_cls, @selector(inMemoryModelWithDescriptor:), desc
    );
    if (!model) {
        set_last_error("ANE fused2 in-memory model creation failed");
        return 0;
    }

    id hx = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString* td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager* fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mil_data writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError* e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e
    );
    if (!ok) {
        set_last_errorf("ANE fused2 compile failed", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    e = nil;
    ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e
    );
    if (!ok) {
        set_last_errorf("ANE fused2 load failed", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    size_t total = (size_t)in_dim * batch +
        (size_t)in_dim * hidden_dim +
        (size_t)hidden_dim +
        (size_t)hidden_dim * out_dim +
        (size_t)out_dim;
    size_t in_bytes = total * sizeof(float);
    size_t r1_bytes = (size_t)batch * hidden_dim * sizeof(float);
    size_t logits_bytes = (size_t)batch * out_dim * sizeof(float);

    IOSurfaceRef in_surface = ane_create_surface(in_bytes);
    IOSurfaceRef r1_surface = ane_create_surface(r1_bytes);
    IOSurfaceRef logits_surface = ane_create_surface(logits_bytes);
    if (!in_surface || !r1_surface || !logits_surface) {
        set_last_error("failed to create ANE fused2 IOSurface");
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    id w_in = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        g_ane_io_cls, @selector(objectWithIOSurface:), in_surface
    );
    id w_r1 = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        g_ane_io_cls, @selector(objectWithIOSurface:), r1_surface
    );
    id w_logits = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        g_ane_io_cls, @selector(objectWithIOSurface:), logits_surface
    );

    id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        g_ane_req_cls,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[w_in], @[@0], @[w_r1, w_logits], @[@0, @1], nil, nil, @0
    );
    if (!request) {
        set_last_error("failed to create ANE fused2 request");
        [fm removeItemAtPath:td error:nil];
        return 0;
    }

    k->used = 1;
    k->batch = batch;
    k->in_dim = in_dim;
    k->hidden_dim = hidden_dim;
    k->out_dim = out_dim;
    k->model = model;
    k->request = request;
    k->in_surface = in_surface;
    k->r1_surface = r1_surface;
    k->logits_surface = logits_surface;
    k->tmp_dir = td;
    return 1;
}

static AneMlp2CacheEntry* ane_get_mlp2_kernel(
    int batch,
    int in_dim,
    int hidden_dim,
    int out_dim
) {
    int hit = ane_find_mlp2_slot(batch, in_dim, hidden_dim, out_dim);
    if (hit >= 0) {
        g_cache_hit_count++;
        return &g_mlp2_cache[hit];
    }

    int slot = -1;
    for (int i = 0; i < ANE_MLP2_CACHE_SIZE; i++) {
        if (!g_mlp2_cache[i].used) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        slot = g_mlp2_evict_cursor % ANE_MLP2_CACHE_SIZE;
        g_mlp2_evict_cursor++;
        ane_free_mlp2_kernel(&g_mlp2_cache[slot]);
    }

    if (!ane_prepare_mlp2_entry(&g_mlp2_cache[slot], batch, in_dim, hidden_dim, out_dim)) {
        ane_free_mlp2_kernel(&g_mlp2_cache[slot]);
        return NULL;
    }

    g_compile_count++;
    return &g_mlp2_cache[slot];
}

static int ane_eval_mlp2(
    AneMlp2CacheEntry* k,
    const float* x,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    float* out_r1,
    float* out_logits
) {
    int batch = k->batch;
    int in_dim = k->in_dim;
    int hidden_dim = k->hidden_dim;
    int out_dim = k->out_dim;

    IOSurfaceLock(k->in_surface, 0, NULL);
    float* inp = (float*)IOSurfaceGetBaseAddress(k->in_surface);
    size_t p = 0;
    for (int c = 0; c < in_dim; c++) {
        for (int s = 0; s < batch; s++) {
            inp[p++] = x[(size_t)s * in_dim + c];
        }
    }
    memcpy(inp + p, w1, (size_t)in_dim * hidden_dim * sizeof(float));
    p += (size_t)in_dim * hidden_dim;
    memcpy(inp + p, b1, (size_t)hidden_dim * sizeof(float));
    p += (size_t)hidden_dim;
    memcpy(inp + p, w2, (size_t)hidden_dim * out_dim * sizeof(float));
    p += (size_t)hidden_dim * out_dim;
    memcpy(inp + p, b2, (size_t)out_dim * sizeof(float));
    IOSurfaceUnlock(k->in_surface, 0, NULL);

    NSError* e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e
    );
    if (!ok) {
        set_last_errorf("ANE fused2 evaluate failed", [[e description] UTF8String]);
        return 0;
    }

    IOSurfaceLock(k->r1_surface, kIOSurfaceLockReadOnly, NULL);
    float* r1_cf = (float*)IOSurfaceGetBaseAddress(k->r1_surface);
    for (int s = 0; s < batch; s++) {
        float* dst = out_r1 + (size_t)s * hidden_dim;
        for (int h = 0; h < hidden_dim; h++) {
            dst[h] = r1_cf[(size_t)h * batch + s];
        }
    }
    IOSurfaceUnlock(k->r1_surface, kIOSurfaceLockReadOnly, NULL);

    IOSurfaceLock(k->logits_surface, kIOSurfaceLockReadOnly, NULL);
    float* lg_cf = (float*)IOSurfaceGetBaseAddress(k->logits_surface);
    for (int s = 0; s < batch; s++) {
        float* dst = out_logits + (size_t)s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            dst[o] = lg_cf[(size_t)o * batch + s];
        }
    }
    IOSurfaceUnlock(k->logits_surface, kIOSurfaceLockReadOnly, NULL);

    return 1;
}

static void cpu_mlp2_forward_cblas(
    const float* x,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    float* out_r1,
    float* out_logits,
    int batch,
    int in_dim,
    int hidden_dim,
    int out_dim
) {
    cpu_dense_relu_cblas(x, w1, b1, out_r1, batch, in_dim, hidden_dim);

    for (int i = 0; i < batch; i++) {
        memcpy(out_logits + (size_t)i * out_dim, b2, (size_t)out_dim * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        batch, out_dim, hidden_dim,
        1.0f,
        out_r1, hidden_dim,
        w2, out_dim,
        1.0f,
        out_logits, out_dim);
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
        if (k && ane_eval_dense(k, x, w, b, out, 1)) {
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

int ane_dense_forward(
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
        set_last_error("invalid ane_dense_forward arguments");
        return ANE_BACKEND_ERROR;
    }

    if (g_opt_in && g_private_framework_found && g_runtime_ready) {
        AnekernelCacheEntry* k = ane_get_kernel(batch, in_dim, out_dim, w);
        if (k && ane_eval_dense(k, x, w, b, out, 0)) {
            set_last_error("ANE eval OK (dense+bias on ANE)");
            return ANE_BACKEND_OK;
        }
    }

    cpu_dense_bias_cblas(x, w, b, out, batch, in_dim, out_dim);
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

int ane_mlp2_forward(
    const float* x,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    float* out_r1,
    float* out_logits,
    int batch,
    int in_dim,
    int hidden_dim,
    int out_dim
) {
    ane_probe_once();

    if (!x || !w1 || !b1 || !w2 || !b2 || !out_r1 || !out_logits ||
        batch <= 0 || in_dim <= 0 || hidden_dim <= 0 || out_dim <= 0) {
        set_last_error("invalid ane_mlp2_forward arguments");
        return ANE_BACKEND_ERROR;
    }

    if (g_opt_in && g_private_framework_found && g_runtime_ready && g_dynamic_mode) {
        AneMlp2CacheEntry* k = ane_get_mlp2_kernel(batch, in_dim, hidden_dim, out_dim);
        if (k && ane_eval_mlp2(k, x, w1, b1, w2, b2, out_r1, out_logits)) {
            set_last_error("ANE eval OK (fused2: r1+logits on ANE)");
            return ANE_BACKEND_OK;
        }
    }

    cpu_mlp2_forward_cblas(
        x, w1, b1, w2, b2, out_r1, out_logits, batch, in_dim, hidden_dim, out_dim
    );
    g_fallback_count++;

    if (g_opt_in && g_private_framework_found && g_runtime_ready && !g_dynamic_mode) {
        set_last_error("ANE fused2 requires ANE_DYNAMIC_WEIGHTS=1; used CPU fallback");
    } else if (g_opt_in && g_private_framework_found && g_runtime_ready) {
        if (strstr(g_last_error, "compile") == NULL &&
            strstr(g_last_error, "evaluate") == NULL &&
            strstr(g_last_error, "descriptor") == NULL) {
            set_last_error("ANE fused2 path failed, used CPU fallback");
        }
    } else if (g_opt_in && g_private_framework_found && !g_runtime_ready) {
        set_last_error("private ANE framework found but required classes unavailable; used CPU fallback");
    } else {
        set_last_error("ANE unavailable: used CPU fallback");
    }

    return ANE_BACKEND_FALLBACK;
}
