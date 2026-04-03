import{Q as e,S as t,X as n,b as r,f as i,i as a,j as o,o as s,s as c,t as l,u}from"./_plugin-vue_export-helper-BytCH4UJ.js";var d={class:`learn-view`},f={class:`learn-layout`},p={class:`section-nav`},m=[`onClick`],h={class:`content-area`},g={class:`content-card`},_={class:`content-title`},v={class:`code-block`},y={class:`content-desc`},b=`typedef struct Tensor {
    Arr *data;          // 前向传播的值
    Arr *grad;          // 反向传播的梯度
    int op;             // 操作类型 (MATMUL, RELU, ...)
    struct Tensor *lhs; // 左操作数
    struct Tensor *rhs; // 右操作数
    Arg arg;            // 操作参数
} Tensor;

typedef struct {
    float *values;      // 实际数据 (64字节对齐)
    int *shape;         // 形状, 如 [1, 784]
    int *strides;       // 步长, 用于索引计算
    int ndim;           // 维度数
    int size;           // 元素总数
} Arr;`,x=`Tensor 是核心数据结构，每个张量保存：
• data: 前向传播的数值
• grad: 反向传播时累积的梯度
• op: 记录产生这个张量的操作（用于反向传播时知道怎么求导）
• lhs/rhs: 计算图的父节点引用

Arr 是底层的多维数组，使用 64 字节对齐分配以优化 SIMD 性能。`,S=`// tensor.h: cblas_sgemm 调用
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    cblas_sgemm(CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        M, N, K,        // 输出行数、列数、缩减维度
        1.0f,           // alpha
        lhs->data->values, lda,
        rhs->data->values, ldb,
        0.0f,           // beta
        result->data->values, ldc);
#endif

// 前向: [1,784] × [784,256] → [1,256]
// C[i,j] = Σ A[i,k] × B[k,j]
//
// 反向传播:
// dL/dA = dL/dC × B^T    → [1,784] = [1,256] × [256,784]
// dL/dB = A^T × dL/dC    → [784,256] = [784,1] × [1,256]`,C=`矩阵乘法是神经网络最核心的操作。

在 macOS 上使用 Apple Accelerate (vDSP/BLAS)，比纯 C 循环快约 26 倍。

反向传播时需要对矩阵乘法求导，利用链式法则：
• 对左操作数的梯度 = 输出梯度 × 右操作数的转置
• 对右操作数的梯度 = 左操作数的转置 × 输出梯度`,w=`void backward(Tensor *t) {
    if (!t || !t->grad) return;

    switch (t->op) {
        case MATMUL: {
            // dL/dA = dL/dC × B^T
            Arr *grad_lhs = matmul_data(t->grad, t->rhs->data, true);
            accumulate_grad(t->lhs, grad_lhs);

            // dL/dB = A^T × dL/dC
            Arr *grad_rhs = matmul_data(t->lhs->data, t->grad, true);
            accumulate_grad(t->rhs, grad_rhs);
            break;
        }
        case RELU: {
            // d(ReLU)/dx = 1 if x > 0 else 0
            for (int i = 0; i < t->lhs->data->size; i++) {
                t->lhs->grad->values[i] +=
                    t->grad->values[i] * (t->lhs->data->values[i] > 0 ? 1 : 0);
            }
            break;
        }
        // ... LOGSOFTMAX, ADD_BIAS, etc.
    }

    // 递归反向传播到父节点
    if (t->lhs) backward(t->lhs);
    if (t->rhs) backward(t->rhs);
}`,T=`自动微分的核心思想：

1. 前向传播时构建计算图（每个 Tensor 记住 op 和父节点）
2. 调用 backward() 时从输出节点开始，按链式法则递归计算梯度
3. 每个操作（MATMUL、RELU、LOGSOFTMAX）都知道自己的局部导数怎么算
4. 梯度通过 accumulate_grad 累加（因为一个节点可能被多次使用）

这和 PyTorch 的 autograd 原理完全一样，只是用 C 手工实现的。`,E=`void kaiming_uniform_(Arr *arr, int fan_in) {
    // Kaiming 均匀初始化 (He initialization)
    // 适用于 ReLU 激活函数
    //
    // limit = sqrt(6 / fan_in)
    // values ~ Uniform(-limit, +limit)

    float limit = sqrtf(6.0f / (float)fan_in);

    for (int i = 0; i < arr->size; i++) {
        arr->values[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
}

// 使用:
// kaiming_uniform_(w1->data, 784);   // fan_in = 784
// kaiming_uniform_(w2->data, 256);   // fan_in = 256`,D=`权重初始化对训练至关重要。

Kaiming 初始化专门为 ReLU 设计：
• 如果权重太大 → 梯度爆炸
• 如果权重太小 → 梯度消失（ReLU 输出全为 0）
• Kaiming 的方差 = 2/fan_in，恰好保持前向和反向的信号方差

对比 Xavier 初始化（适用于 tanh/sigmoid）：
• Xavier 方差 = 1/fan_in
• Kaiming 方差 = 2/fan_in（因为 ReLU 把负值截断了）`,O=`// train.c 核心训练循环
for (int step = 0; step < TRAIN_STEPS; step++) {
    // 1. 采样一个 batch
    int *indices = sampler_next(sampler);

    // 2. 前向传播
    Tensor *x = tensor_from_arr(batch_x);
    Tensor *h1 = matmul(x, w1);
    Tensor *h1b = add_bias(h1, b1);
    Tensor *r1 = relu(h1b);
    Tensor *h2 = matmul(r1, w2);
    Tensor *h2b = add_bias(h2, b2);
    Tensor *out = logsoftmax(h2b);

    // 3. 计算 NLL Loss
    Tensor *loss = nll_loss(out, batch_y);

    // 4. 反向传播
    backward(loss);

    // 5. SGD 更新权重
    for (int i = 0; i < w1->data->size; i++) {
        w1->data->values[i] -= lr * w1->grad->values[i];
    }
    // ... b1, w2, b2 同理

    // 6. 清零梯度
    zero_grad(w1); zero_grad(b1);
    zero_grad(w2); zero_grad(b2);

    // 7. 释放中间张量（避免内存泄漏）
    free_tensor(x); free_tensor(h1); ...
}`,k=`训练循环遵循标准的深度学习流程：

1. 前向传播：输入经过矩阵乘法、激活函数得到输出
2. 计算损失：NLL Loss（负对数似然），等价于 CrossEntropy
3. 反向传播：从 loss 开始 backward()，自动计算所有参数的梯度
4. 参数更新：SGD，w = w - lr * grad
5. 清零梯度：每次迭代后必须清零，否则梯度会累积
6. 释放中间张量：C 语言没有 GC，需要手工管理内存

关键参数：
• batch_size = 128
• learning_rate = 0.005
• train_steps = 20,000
• hidden_size = 256`,A=`// 64 字节对齐分配（优化 SIMD）
void* aligned_alloc_64(size_t size) {
    void *ptr;
    if (posix_memalign(&ptr, 64, size) != 0) {
        perror("posix_memalign");
        exit(1);
    }
    return ptr;
}

// 创建张量时分配所有需要的内存
Tensor* create_zero_tensor(int *shape, int ndim) {
    Tensor *t = malloc(sizeof(Tensor));
    t->data = malloc(sizeof(Arr));
    t->data->ndim = ndim;
    t->data->shape = malloc(ndim * sizeof(int));
    memcpy(t->data->shape, shape, ndim * sizeof(int));
    t->data->size = 1;
    for (int i = 0; i < ndim; i++)
        t->data->size *= shape[i];
    t->data->values = aligned_alloc_64(t->data->size * sizeof(float));
    memset(t->data->values, 0, t->data->size * sizeof(float));
    return t;
}

// 每个前向操作创建的中间 Tensor 必须手动释放
free_tensor(x);    // 输入
free_tensor(h1);   // matmul 输出
free_tensor(h1b);  // add_bias 输出
free_tensor(r1);   // relu 输出`,j=`C 语言没有垃圾回收，内存管理是必须注意的问题。

关键设计：
• 64 字节对齐：SIMD 指令（如 AVX-512）要求数据对齐
• posix_memalign：跨平台的对齐分配方式
• 每个 Tensor 操作都创建新的 Tensor，旧的需要释放
• 训练循环中每个 step 产生 ~7 个中间 Tensor，20000 步就是 14 万个

内存泄漏排查技巧：
• valgrind --leak-check=full ./train
• 确保每个 create_zero_tensor 都有对应的 free_tensor`,M=l({__name:`LearnView`,setup(l){let M=o(`tensor`),N=[{id:`tensor`,label:`Tensor 结构`},{id:`matmul`,label:`矩阵乘法`},{id:`autograd`,label:`自动微分`},{id:`init`,label:`权重初始化`},{id:`train`,label:`训练循环`},{id:`memory`,label:`内存管理`}],P=s(()=>({tensor:{code:b,desc:x,title:`Tensor 数据结构`},matmul:{code:S,desc:C,title:`矩阵乘法 (cblas_sgemm)`},autograd:{code:w,desc:T,title:`自动微分 (backward)`},init:{code:E,desc:D,title:`Kaiming 权重初始化`},train:{code:O,desc:k,title:`训练循环`},memory:{code:A,desc:j,title:`内存管理`}})[M.value]);return(o,s)=>(r(),u(`div`,d,[s[0]||=c(`header`,{class:`page-header`},[c(`h1`,null,`⚙️ C 代码原理`),c(`p`,{class:`page-desc`},`深入 tensor.h 和 train.c 的实现细节`)],-1),c(`div`,f,[c(`nav`,p,[(r(),u(a,null,t(N,t=>c(`button`,{key:t.id,class:n([`nav-item`,{active:M.value===t.id}]),onClick:e=>M.value=t.id},e(t.label),11,m)),64))]),c(`main`,h,[c(`div`,g,[c(`h2`,_,e(P.value.title),1),c(`div`,v,[c(`pre`,null,[c(`code`,null,e(P.value.code),1)])]),c(`div`,y,[(r(!0),u(a,null,t(P.value.desc.split(`

`),(t,n)=>(r(),u(`p`,{key:n},e(t),1))),128))])])])]),s[1]||=i(`<div class="project-structure" data-v-6c941080><h2 data-v-6c941080>项目文件结构</h2><div class="file-tree" data-v-6c941080><div class="file-item" data-v-6c941080><span class="file-icon" data-v-6c941080>📄</span> <span class="file-name" data-v-6c941080>tensor.h</span> <span class="file-desc" data-v-6c941080>核心张量库，header-only，包含 Tensor 结构、矩阵运算、自动微分</span></div><div class="file-item" data-v-6c941080><span class="file-icon" data-v-6c941080>📄</span> <span class="file-name" data-v-6c941080>tensor_web.h</span> <span class="file-desc" data-v-6c941080>Web 服务辅助函数，前向传播追踪、JSON 序列化</span></div><div class="file-item" data-v-6c941080><span class="file-icon" data-v-6c941080>📄</span> <span class="file-name" data-v-6c941080>train.c</span> <span class="file-desc" data-v-6c941080>训练程序：加载 CSV → 训练 20000 步 → 保存 mnist_mlp.bin</span></div><div class="file-item" data-v-6c941080><span class="file-icon" data-v-6c941080>📄</span> <span class="file-name" data-v-6c941080>eval.c</span> <span class="file-desc" data-v-6c941080>评估程序：加载模型 → 在 10000 个测试样本上计算准确率</span></div><div class="file-item" data-v-6c941080><span class="file-icon" data-v-6c941080>📄</span> <span class="file-name" data-v-6c941080>web_server.c</span> <span class="file-desc" data-v-6c941080>HTTP 服务器：纯 C 实现，端口 3000，提供 REST API</span></div><div class="file-item" data-v-6c941080><span class="file-icon" data-v-6c941080>📄</span> <span class="file-name" data-v-6c941080>api_handlers.c</span> <span class="file-desc" data-v-6c941080>API 处理器：/api/predict、/api/eval、/api/architecture</span></div><div class="file-item" data-v-6c941080><span class="file-icon" data-v-6c941080>📄</span> <span class="file-name" data-v-6c941080>Makefile</span> <span class="file-desc" data-v-6c941080>构建配置：macOS 用 Accelerate，Linux 用 OpenMP</span></div></div></div>`,1)]))}},[[`__scopeId`,`data-v-6c941080`]]);export{M as default};