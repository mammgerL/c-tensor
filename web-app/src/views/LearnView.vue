<script setup>
import { ref, computed } from 'vue'

const activeSection = ref('tensor')
const trainStep = ref(0)

const sections = [
  { id: 'tensor', label: 'Tensor 结构' },
  { id: 'matmul', label: '矩阵乘法' },
  { id: 'autograd', label: '自动微分' },
  { id: 'init', label: '权重初始化' },
  { id: 'train', label: '训练循环' },
  { id: 'memory', label: '内存管理' },
]

const tensorCode = `typedef struct Tensor {
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
} Arr;`

const tensorDesc = `Tensor 是核心数据结构，每个张量保存：
• data: 前向传播的数值
• grad: 反向传播时累积的梯度
• op: 记录产生这个张量的操作（用于反向传播时知道怎么求导）
• lhs/rhs: 计算图的父节点引用

Arr 是底层的多维数组，使用 64 字节对齐分配以优化 SIMD 性能。`

const matmulCode = `// tensor.h: cblas_sgemm 调用
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
// dL/dB = A^T × dL/dC    → [784,256] = [784,1] × [1,256]`

const matmulDesc = `矩阵乘法是神经网络最核心的操作。

在 macOS 上使用 Apple Accelerate (vDSP/BLAS)，比纯 C 循环快约 26 倍。

反向传播时需要对矩阵乘法求导，利用链式法则：
• 对左操作数的梯度 = 输出梯度 × 右操作数的转置
• 对右操作数的梯度 = 左操作数的转置 × 输出梯度`

const autogradCode = `void backward(Tensor *t) {
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
}`

const autogradDesc = `自动微分的核心思想：

1. 前向传播时构建计算图（每个 Tensor 记住 op 和父节点）
2. 调用 backward() 时从输出节点开始，按链式法则递归计算梯度
3. 每个操作（MATMUL、RELU、LOGSOFTMAX）都知道自己的局部导数怎么算
4. 梯度通过 accumulate_grad 累加（因为一个节点可能被多次使用）

这和 PyTorch 的 autograd 原理完全一样，只是用 C 手工实现的。`

const initCode = `void kaiming_uniform_(Arr *arr, int fan_in) {
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
// kaiming_uniform_(w2->data, 256);   // fan_in = 256`

const initDesc = `权重初始化对训练至关重要。

Kaiming 初始化专门为 ReLU 设计：
• 如果权重太大 → 梯度爆炸
• 如果权重太小 → 梯度消失（ReLU 输出全为 0）
• Kaiming 的方差 = 2/fan_in，恰好保持前向和反向的信号方差

对比 Xavier 初始化（适用于 tanh/sigmoid）：
• Xavier 方差 = 1/fan_in
• Kaiming 方差 = 2/fan_in（因为 ReLU 把负值截断了）`

const trainCode = `// train.c 核心训练循环
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
}`

const trainDesc = `训练循环遵循标准的深度学习流程：

1. 前向传播：输入经过矩阵乘法、激活函数得到输出。
2. 计算损失：NLL Loss（负对数似然），衡量预测值与真实标签的差距。
3. 反向传播：从 loss 开始 backward()，利用链式法则自动计算所有参数的梯度。
4. 参数更新：SGD，w = w - lr * grad，沿着梯度下降方向微调权重。
5. 清零梯度：每次迭代后必须清零，否则梯度会累积。
6. 释放中间张量：C 语言没有 GC，需要手工管理内存。

关键参数：
• batch_size = 128
• learning_rate = 0.005
• train_steps = 20,000
• hidden_size = 256`

const memoryCode = `// 64 字节对齐分配（优化 SIMD）
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
free_tensor(r1);   // relu 输出`

const memoryDesc = `C 语言没有垃圾回收，内存管理是必须注意的问题。

关键设计：
• 64 字节对齐：SIMD 指令（如 AVX-512）要求数据对齐
• posix_memalign：跨平台的对齐分配方式
• 每个 Tensor 操作都创建新的 Tensor，旧的需要释放
• 训练循环中每个 step 产生 ~7 个中间 Tensor，20000 步就是 14 万个

内存泄漏排查技巧：
• valgrind --leak-check=full ./train
• 确保每个 create_zero_tensor 都有对应的 free_tensor`

const sectionContent = computed(() => {
  const map = {
    tensor: { code: tensorCode, desc: tensorDesc, title: 'Tensor 数据结构' },
    matmul: { code: matmulCode, desc: matmulDesc, title: '矩阵乘法 (cblas_sgemm)' },
    autograd: { code: autogradCode, desc: autogradDesc, title: '自动微分 (backward)' },
    init: { code: initCode, desc: initDesc, title: 'Kaiming 权重初始化' },
    train: { code: trainCode, desc: trainDesc, title: '训练循环' },
    memory: { code: memoryCode, desc: memoryDesc, title: '内存管理' },
  }
  return map[activeSection.value]
})
</script>

<template>
  <div class="learn-view">
    <header class="page-header">
      <h1>⚙️ C 代码原理</h1>
      <p class="page-desc">深入 tensor.h 和 train.c 的实现细节</p>
    </header>

    <div class="learn-layout">
      <nav class="section-nav">
        <button
          v-for="sec in sections"
          :key="sec.id"
          :class="['nav-item', { active: activeSection === sec.id }]"
          @click="activeSection = sec.id"
        >
          {{ sec.label }}
        </button>
      </nav>

       <main class="content-area">
         <div class="content-card">
           <h2 class="content-title">{{ sectionContent.title }}</h2>
           
           <div v-if="activeSection === 'train'" class="train-visualizer">
             <div class="step-controls">
               <button 
                 v-for="(step, i) in [
                   { label: '前向传播', desc: '计算预测值' },
                   { label: '计算损失', desc: '衡量误差' },
                   { label: '反向传播', desc: '计算梯度' },
                   { label: '参数更新', desc: '优化权重' }
                 ]" 
                 :key="i"
                 :class="['step-btn', { active: trainStep === i }]"
                 @click="trainStep = i"
               >
                 <span class="step-num">{{ i + 1 }}</span>
                 <span class="step-label">{{ step.label }}</span>
               </button>
             </div>

             <div class="viz-canvas">
               <div class="network-diagram">
                 <div class="node input-node" :class="{ active: trainStep === 0 || trainStep === 2 }">输入</div>
                 <div class="arrow forward" :class="{ active: trainStep === 0 }">→</div>
                 <div class="node hidden-node" :class="{ active: trainStep === 0 || trainStep === 2 }">隐藏层</div>
                 <div class="arrow forward" :class="{ active: trainStep === 0 }">→</div>
                 <div class="node output-node" :class="{ active: trainStep === 1 }">输出/Loss</div>
                 <div class="arrow backward" :class="{ active: trainStep === 2 }">←</div>
                 <div class="node hidden-node" :class="{ active: trainStep === 2 }">隐藏层</div>
                 <div class="arrow backward" :class="{ active: trainStep === 2 }">←</div>
                 <div class="node input-node" :class="{ active: trainStep === 2 }">输入</div>
                 <div class="update-flash" :class="{ active: trainStep === 3 }">权重更新 ⚡</div>
               </div>
               <div class="step-info">
                 <p v-if="trainStep === 0"><b>前向传播 (Forward):</b> 数据流从输入层 → 隐藏层 → 输出层。执行 $\text{ReLU}(XW_1 + b_1)$ 和 $\text{Softmax}(HW_2 + b_2)$。</p>
                 <p v-if="trainStep === 1"><b>计算损失 (Loss):</b> 将输出与真实标签对比。Loss 越大，表示模型预测越不准确。</p>
                 <p v-if="trainStep === 2"><b>反向传播 (Backward):</b> 误差信号反向流动，计算每个权重对 Loss 的贡献（梯度 $\partial L / \partial W$）。</p>
                 <p v-if="trainStep === 3"><b>参数更新 (Update):</b> 根据梯度方向，稍微减小权重值：$W_{new} = W_{old} - \eta \cdot \text{grad}$。</p>
               </div>
             </div>
           </div>

           <div class="code-block">
             <pre><code>{{ sectionContent.code }}</code></pre>
           </div>
           <div class="content-desc">
             <p v-for="(para, i) in sectionContent.desc.split('\n\n')" :key="i">{{ para }}</p>
           </div>
         </div>
       </main>
    </div>

    <div class="project-structure">
      <h2>项目文件结构</h2>
      <div class="file-tree">
        <div class="file-item"><span class="file-icon">📄</span> <span class="file-name">tensor.h</span> <span class="file-desc">核心张量库，header-only，包含 Tensor 结构、矩阵运算、自动微分</span></div>
        <div class="file-item"><span class="file-icon">📄</span> <span class="file-name">train.c</span> <span class="file-desc">训练程序：加载 CSV → 训练 20000 步 → 保存 mnist_mlp.bin</span></div>
        <div class="file-item"><span class="file-icon">📄</span> <span class="file-name">eval.c</span> <span class="file-desc">评估程序：加载模型 → 在 10000 个测试样本上计算准确率</span></div>
        <div class="file-item"><span class="file-icon">📄</span> <span class="file-name">Makefile</span> <span class="file-desc">构建配置：macOS 用 Accelerate，Linux 用 OpenMP</span></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.learn-view {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 24px;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.page-header h1 {
  font-size: 32px;
  font-weight: 800;
  margin-bottom: 8px;
}

.page-desc {
  font-size: 16px;
  color: var(--color-text-light);
}

.learn-layout {
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 24px;
  margin-bottom: 48px;
}

.section-nav {
  display: flex;
  flex-direction: column;
  gap: 4px;
  position: sticky;
  top: 80px;
  align-self: start;
}

.nav-item {
  padding: 10px 16px;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 600;
  text-align: left;
  background: transparent;
  color: var(--color-text-light);
}

.nav-item:hover {
  background: var(--color-bg);
  color: var(--color-text);
}

.nav-item.active {
  background: var(--color-primary);
  color: white;
}

.content-card {
  background: var(--color-card);
  border-radius: 16px;
  padding: 28px;
  box-shadow: var(--shadow-sm);
}

.content-title {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--color-border);
}

.code-block {
  background: #1e1e2e;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  overflow-x: auto;
}

.code-block pre {
  margin: 0;
  font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 13px;
  line-height: 1.6;
  color: #cdd6f4;
  white-space: pre;
}

.content-desc {
  font-size: 14px;
  line-height: 1.8;
  color: var(--color-text-light);
}

.content-desc p {
   margin-bottom: 12px;
}

.train-visualizer {
  margin-bottom: 32px;
  padding: 24px;
  background: var(--color-bg);
  border-radius: 16px;
  border: 1px solid var(--color-border);
}

.step-controls {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-bottom: 32px;
}

.step-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 20px;
  border: 1px solid var(--color-border);
  background: var(--color-card);
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
  color: var(--color-text-light);
}

.step-btn.active {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
  box-shadow: 0 4px 12px rgba(var(--color-primary-rgb), 0.3);
}

.step-num {
  background: rgba(0,0,0,0.1);
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
}

.step-btn.active .step-num {
  background: rgba(255,255,255,0.2);
}

.viz-canvas {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
}

.network-diagram {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 40px;
  position: relative;
}

.node {
  padding: 12px 20px;
  border-radius: 8px;
  background: var(--color-card);
  border: 2px solid var(--color-border);
  font-size: 14px;
  font-weight: 600;
  transition: all 0.3s;
  z-index: 2;
}

.node.active {
  border-color: var(--color-primary);
  color: var(--color-primary);
  transform: scale(1.1);
  box-shadow: 0 0 15px rgba(var(--color-primary-rgb), 0.4);
}

.arrow {
  font-size: 24px;
  font-weight: bold;
  color: var(--color-border);
  transition: all 0.3s;
}

.arrow.active {
  color: var(--color-primary);
  transform: scale(1.3);
}

.update-flash {
  position: absolute;
  top: -20px;
  padding: 4px 12px;
  background: #facc15;
  color: #854d0e;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  opacity: 0;
  transition: all 0.3s;
}

.update-flash.active {
  opacity: 1;
  top: 10px;
  animation: flash 1s infinite;
}

@keyframes flash {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.step-info {
  max-width: 600px;
  text-align: center;
  font-size: 15px;
  line-height: 1.6;
  color: var(--color-text-light);
  min-height: 60px;
}

@media (max-width: 768px) {
  .network-diagram {
    flex-wrap: wrap;
    gap: 8px;
  }
}

.project-structure {
  background: var(--color-card);
  border-radius: 16px;
  padding: 28px;
  box-shadow: var(--shadow-sm);
}

.project-structure h2 {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 20px;
}

.file-tree {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  background: var(--color-bg);
  border-radius: 10px;
  font-size: 14px;
}

.file-icon {
  font-size: 18px;
}

.file-name {
  font-weight: 600;
  min-width: 130px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
}

.file-desc {
  color: var(--color-text-light);
  font-size: 13px;
}

@media (max-width: 768px) {
  .learn-layout {
    grid-template-columns: 1fr;
  }

  .section-nav {
    position: static;
    flex-direction: row;
    flex-wrap: wrap;
  }
}
</style>
