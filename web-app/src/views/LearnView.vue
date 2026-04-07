<script setup>
import { computed, ref } from 'vue'

const activeSection = ref('architecture')

const sections = [
  {
    id: 'architecture',
    label: '整体结构',
    title: '仓库怎么把“训练一个 MLP”拆成几个 C 文件',
    intro: '这个项目并不是一个“大框架”，而是把最少的几件事拆干净：`tensor.h` 负责张量和自动微分，`train.c` 负责数据加载与训练循环，`eval.c` 负责离线评估。',
    trainCode: {
      file: 'train.c',
      lines: '176-208',
      title: 'model + hyperparams',
      code: `Tensor* x = create_zero_tensor((int[]) { N, 784 }, 2);
Tensor* y = create_zero_tensor((int[]) { N, 10 }, 2);

int H = 256;
Tensor* w1 = create_zero_tensor((int[]) { 784, H }, 2);
Tensor* b1 = create_zero_tensor((int[]) { H }, 1);
Tensor* w2 = create_zero_tensor((int[]) { H, 10 }, 2);
Tensor* b2 = create_zero_tensor((int[]) { 10 }, 1);

int B = 128;
float lr = 0.005f;
int steps = 20000;`,
    },
    tensorCode: {
      file: 'tensor.h',
      lines: '34-41',
      title: 'op code 约定',
      code: `#define MATMUL     0
#define MEAN       1
#define MUL        2
#define RELU       3
#define LOGSOFTMAX 4
#define SUM_AXIS1  5
#define ADD_BIAS   6`,
    },
    bullets: [
      { title: '模型非常小', text: '网络结构就是 `784 → 256 → 10` 两层全连接。这样代码足够短，适合把训练全流程写透。' },
      { title: 'header-only 的代价与收益', text: '`tensor.h` 把算子、梯度、内存分配都内联在一个头文件里，调用简单，但也要求每个实现都足够克制。' },
      { title: '训练和推理分离', text: '`train.c` 只负责产生 `mnist_mlp.bin`；网页与 `eval.c` 都消费模型文件，而不是依赖训练时状态。' },
    ],
  },
  {
    id: 'tensor',
    label: 'Tensor',
    title: 'Tensor / Arr：值、梯度、前驱节点组成最小计算图单元',
    intro: '`Arr` 负责真正存储多维数组，`Tensor` 在它之上再叠一层 autograd 元数据：这个值是哪个 op 产生的、它依赖谁、它的梯度要累加到哪里。',
    trainCode: {
      file: 'tensor.h',
      lines: '43-64',
      title: 'Arr + Tensor struct',
      code: `typedef struct {
    float* values;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;

typedef struct Tensor {
    Arr* data;
    Arr* grad;
    int op;
    struct Tensor* prevs[MAX_PREVS];
    int num_prevs;
    Arg args[MAX_ARGS];
} Tensor;`,
    },
    tensorCode: {
      file: 'tensor.h',
      lines: '98-125,153-160',
      title: 'shape / strides / zero init',
      code: `arr->size = 1;
for (int i = ndim - 1; i >= 0; i--) {
    arr->strides[i] = arr->size;
    arr->size *= shape[i];
}

size_t bytes = (size_t)arr->size * sizeof(float);
arr->values = (float*)aligned_alloc_64(bytes);
memset(arr->values, 0, bytes);

Tensor* t = (Tensor*)malloc(sizeof(Tensor));
t->data = d;
t->grad = create_arr_zeros(shape, ndim);`,
    },
    bullets: [
      { title: '为什么同时存 `data` 和 `grad`', text: '前向传播需要读 `data`，反向传播需要往 `grad` 里累加。把二者并排放在一个 Tensor 上，代码简单而且容易定位。' },
      { title: '为什么有 `prevs`', text: '这就是计算图的边。每个算子把自己的输入记为前驱，`backward()` 只要递归遍历它们，就能回溯整张图。' },
      { title: '为什么要 `strides`', text: '代码里虽然大多数张量是连续二维数组，但 `shape + strides` 让索引逻辑统一，`sum_axis1`、`add_bias` 等算子都能按通用方式取值。' },
    ],
  },
  {
    id: 'ops',
    label: '核心算子',
    title: '前向算子：matmul、relu、logsoftmax 负责把像素一路变成对数概率',
    intro: '网页里看到的“输入 784 像素 → 256 维隐藏层 → 10 类输出”，在 C 里其实只依赖几种非常原子的算子。每个算子都同时承担两件事：算出前向值，登记自己的前驱。',
    trainCode: {
      file: 'tensor.h',
      lines: '415-451',
      title: 'matmul forward',
      code: `int P = a->data->shape[0];
int Q = a->data->shape[1];
int R = b->data->shape[1];
Tensor* t = create_zero_tensor((int[]) { P, R }, 2);

#if USE_ACCELERATE
vDSP_mmul(a->data->values, 1,
    b->data->values, 1,
    t->data->values, 1,
    (vDSP_Length)P, (vDSP_Length)R, (vDSP_Length)Q);
#else
for (int i = 0; i < P; i++) { ... }
#endif

t->op = MATMUL;
t->prevs[0] = a;
t->prevs[1] = b;`,
    },
    tensorCode: {
      file: 'tensor.h',
      lines: '290-337',
      title: 'relu + logsoftmax',
      code: `for (int i = 0; i < inp->data->size; i++)
    t->data->values[i] =
        (inp->data->values[i] > 0) ? inp->data->values[i] : 0.0f;

float maxv = inp->data->values[base];
for (int c = 1; c < C; c++) {
    float v = inp->data->values[base + c * inp->data->strides[1]];
    if (v > maxv) maxv = v;
}
float sumexp = 0.0f;
for (int c = 0; c < C; c++)
    sumexp += expf(v - maxv);
t->data->values[pos] =
    inp->data->values[pos] - maxv - logf(sumexp);`,
    },
    bullets: [
      { title: 'matmul 是主力算子', text: '隐藏层和输出层都靠矩阵乘法完成。这个项目把优化预算主要放在这里，所以 macOS 直接走 Accelerate 的 `vDSP_mmul`。' },
      { title: 'ReLU 极简但关键', text: '它只做一件事：把负值清零。没有这一步，两层线性变换仍可折叠成一层线性层。' },
      { title: 'LogSoftmax 先减最大值', text: '这是典型的数值稳定技巧。先做 `x - max` 再算 `exp`，能避免指数溢出。' },
    ],
  },
  {
    id: 'autograd',
    label: '自动微分',
    title: 'backward() 不负责“懂数学”，它负责调度每个 op 的局部梯度公式',
    intro: '很多人第一次看这个项目会以为 `backward()` 很复杂，实际上它很短。真正的关键不在调度器本身，而在每个算子都各自实现了自己的局部 backward 公式。',
    trainCode: {
      file: 'tensor.h',
      lines: '211-220',
      title: 'dispatcher',
      code: `static inline void backward(Tensor* t) {
    if (t->op == MUL) mul_backward(t);
    else if (t->op == MEAN) mean_backward(t);
    else if (t->op == MATMUL) matmul_backward(t);
    else if (t->op == RELU) relu_backward(t);
    else if (t->op == LOGSOFTMAX) logsoftmax_backward(t);
    else if (t->op == SUM_AXIS1) sum_axis1_backward(t);
    else if (t->op == ADD_BIAS) add_bias_backward(t);

    for (int i = 0; i < t->num_prevs; i++)
        backward(t->prevs[i]);
}`,
    },
    tensorCode: {
      file: 'tensor.h',
      lines: '454-492',
      title: 'matmul backward',
      code: `// dA += dC * B^T
cblas_sgemm(CblasRowMajor,
    CblasNoTrans, CblasTrans,
    P, Q, R,
    1.0f,
    out->grad->values, R,
    b->data->values, R,
    1.0f,
    a->grad->values, Q);

// dB += A^T * dC
cblas_sgemm(CblasRowMajor,
    CblasTrans, CblasNoTrans,
    Q, R, P,
    1.0f,
    a->data->values, Q,
    out->grad->values, R,
    1.0f,
    b->grad->values, R);`,
    },
    bullets: [
      { title: '调度和数学分离', text: '`backward()` 只做“根据 op 分发 + 递归到前驱”，真正的链式法则细节藏在 `relu_backward`、`matmul_backward`、`logsoftmax_backward` 里。' },
      { title: '梯度是累加的', text: '所有 backward 实现都用 `+=`，这说明同一个节点可以从多条路径收到梯度，设计上已经对 DAG 做了准备。' },
      { title: '从标量 loss 出发', text: '训练时先手动写入 `loss->grad->values[0] = 1.0f`，相当于把 `dL/dL = 1` 放进图里。' },
    ],
  },
  {
    id: 'training',
    label: '训练循环',
    title: '训练循环本质上就是：采样、前向、损失、反向、更新、清理',
    intro: '`train.c` 把整个深度学习流程写成了一个几乎可以逐行朗读的 for-loop。它没有隐藏框架魔法，所以很适合拿来理解“训练”到底发生了什么。',
    trainCode: {
      file: 'train.c',
      lines: '228-286',
      title: 'one training step',
      code: `for (int it = 0; it < steps; it++) {
    get_next_batch(&sampler, batch_x, batch_y, x, y, B);

    Tensor* h1 = matmul(batch_x, w1);
    Tensor* h1b = add_bias(h1, b1);
    Tensor* r1 = relu(h1b);
    Tensor* h2 = matmul(r1, w2);
    Tensor* h2b = add_bias(h2, b2);
    Tensor* lout = logsoftmax(h2b);

    Tensor* mul_out = mul(lout, batch_y);
    Tensor* per_sample = sum_axis1(mul_out);
    Tensor* loss = mean(per_sample);

    loss->grad->values[0] = 1.0f;
    backward(loss);
    // SGD update + free intermediates
}`,
    },
    tensorCode: {
      file: 'train.c',
      lines: '112-166',
      title: 'batch sampler',
      code: `static void sampler_shuffle(BatchSampler* s) {
    for (int i = s->N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = s->idx[i];
        s->idx[i] = s->idx[j];
        s->idx[j] = tmp;
    }
    s->pos = 0;
}

for (int i = 0; i < B; i++) {
    int idx = s->idx[base + i];
    memcpy(bx + (size_t)i * D, X + (size_t)idx * D, ...);
    memcpy(by + (size_t)i * C, Y + (size_t)idx * C, ...);
}`,
    },
    bullets: [
      { title: 'loss 并不是黑盒函数', text: '项目没有调用一个现成的 `nll_loss`，而是用 `mul + sum_axis1 + mean` 把它拆成了 3 个可求导原子操作。' },
      { title: 'batch sampler 很朴素但高效', text: '先 Fisher-Yates 打乱索引，再连续取 `B=128` 个，用完就重洗。对小项目来说，这比复杂 dataloader 更直接。' },
      { title: '训练 loop 里顺便做性能统计', text: '每 200 step 打印一次 loss 和 step 时间，所以这个文件既是训练逻辑，也是性能观察入口。' },
    ],
  },
  {
    id: 'performance',
    label: '性能与内存',
    title: '为什么这份纯 C 代码还能跑得像样：对齐、BLAS、手动释放',
    intro: '这个仓库的优化重点不是“炫技 SIMD”，而是把最贵的几处路径处理好：连续内存、64 字节对齐、矩阵乘法走平台库，以及训练循环里主动 free 中间张量。',
    trainCode: {
      file: 'tensor.h',
      lines: '88-125,163-167',
      title: 'aligned alloc + free',
      code: `static inline void* aligned_alloc_64(size_t bytes) {
    void* p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) p = NULL;
    return p;
}

arr->values = (float*)aligned_alloc_64(bytes);
memset(arr->values, 0, bytes);

static inline void free_tensor(Tensor* t) {
    if (!t) return;
    if (t->data) free_arr(t->data);
    if (t->grad) free_arr(t->grad);
    free(t);
}`,
    },
    tensorCode: {
      file: 'tensor.h / train.c',
      lines: '78-82,198-205,255-286',
      title: 'kaiming + SGD + cleanup',
      code: `static inline float kaiming_uniform(int fan_in) {
    float gain = sqrtf(2.0f);
    float std = gain / sqrtf((float)fan_in);
    float bound = sqrtf(3.0f) * std;
    return rand_range(-bound, bound);
}

for (int i = 0; i < w1->data->size; i++)
    w1->data->values[i] = kaiming_uniform(784);

for (int i = 0; i < sz; i++) {
    w[i] -= g[i] * lr;
    g[i] = 0.0f;
}
free_tensor(h1);
free_tensor(h1b);`,
    },
    bullets: [
      { title: '64 字节对齐并不花哨', text: '它的目的很直接：给底层 SIMD / BLAS 更规整的内存布局，减少未对齐访问带来的惩罚。' },
      { title: 'Kaiming 初始化是训练稳定性的第一道保险', text: '它让 ReLU 网络在一开始既不全爆炸，也不全塌成 0。' },
      { title: '内存回收是训练正确性的一部分', text: '如果中间 Tensor 不释放，这个 loop 不是“慢一点”，而是会在长跑时稳定泄漏。对 C 来说，这属于逻辑正确性，而不只是风格问题。' },
    ],
  },
]

const currentSectionData = computed(() => sections.find((section) => section.id === activeSection.value) || sections[0])

function goToSection(sectionId) {
  activeSection.value = sectionId
}
</script>

<template>
  <div class="learn-view">
    <header class="page-header">
      <h1>C 代码原理</h1>
      <p class="page-desc">把 `tensor.h` 和 `train.c` 拆开看，理解这个纯 C 训练框架到底是怎么工作的。</p>
    </header>

    <section class="hero-panel">
      <div class="hero-copy">
        <div class="section-kicker">从代码看原理</div>
        <h2>不是“讲概念后贴代码”，而是让概念直接长在真实实现上</h2>
        <p>
          这页聚焦 6 个核心主题：数据结构、前向算子、自动微分、训练循环，以及背后的性能与内存约束。
          目标不是把所有 C 语法解释一遍，而是帮你建立“这几百行代码为什么能训练一个模型”的整体理解。
        </p>
      </div>

      <div class="hero-metrics">
        <div class="metric-card">
          <span class="metric-value">784 → 256 → 10</span>
          <span class="metric-label">网络结构</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">7</span>
          <span class="metric-label">核心算子</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">20000</span>
          <span class="metric-label">训练步数</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">header-only</span>
          <span class="metric-label">`tensor.h` 风格</span>
        </div>
      </div>
    </section>

    <div class="learn-layout">
      <nav class="section-nav">
        <button
          v-for="section in sections"
          :key="section.id"
          :class="['nav-item', { active: activeSection === section.id }]"
          @click="goToSection(section.id)"
        >
          <span class="nav-label">{{ section.label }}</span>
          <span class="nav-title">{{ section.title }}</span>
        </button>
      </nav>

      <main class="content-area">
        <article class="content-card">
          <div class="content-head">
            <div class="section-kicker">{{ currentSectionData.label }}</div>
            <h2 class="content-title">{{ currentSectionData.title }}</h2>
            <p class="content-intro">{{ currentSectionData.intro }}</p>
          </div>

          <div class="visual-card">
            <div v-if="currentSectionData.id === 'architecture'" class="viz-architecture">
              <div class="file-lane">
                <div class="file-box">
                  <span class="file-name">tensor.h</span>
                  <span class="file-role">张量、算子、autograd</span>
                </div>
                <div class="file-arrow">→</div>
                <div class="file-box accent">
                  <span class="file-name">train.c</span>
                  <span class="file-role">采样、训练、保存模型</span>
                </div>
                <div class="file-arrow">→</div>
                <div class="file-box success">
                  <span class="file-name">mnist_mlp.bin</span>
                  <span class="file-role">网页与 eval 共享模型</span>
                </div>
              </div>
              <div class="architecture-notes">
                <div class="note-chip">训练代码只负责产出权重文件</div>
                <div class="note-chip">网页推理复用同一份二进制模型</div>
                <div class="note-chip">核心数学都在 `tensor.h` 里</div>
              </div>
            </div>

            <div v-else-if="currentSectionData.id === 'tensor'" class="viz-tensor">
              <div class="tensor-stack">
                <div class="tensor-panel top">
                  <span class="panel-title">Tensor</span>
                  <div class="tensor-row"><span>data</span><span>前向值</span></div>
                  <div class="tensor-row"><span>grad</span><span>累计梯度</span></div>
                  <div class="tensor-row"><span>op</span><span>产生它的算子</span></div>
                  <div class="tensor-row"><span>prevs</span><span>前驱节点列表</span></div>
                </div>
                <div class="stack-arrow">↓</div>
                <div class="tensor-panel bottom">
                  <span class="panel-title">Arr</span>
                  <div class="tensor-row"><span>values</span><span>连续 float 内存</span></div>
                  <div class="tensor-row"><span>shape</span><span>维度</span></div>
                  <div class="tensor-row"><span>strides</span><span>索引步长</span></div>
                  <div class="tensor-row"><span>size</span><span>总元素个数</span></div>
                </div>
              </div>
            </div>

            <div v-else-if="currentSectionData.id === 'ops'" class="viz-ops">
              <div class="ops-flow">
                <div class="ops-box input">x (128, 784)</div>
                <div class="ops-arrow">× W₁</div>
                <div class="ops-box hidden">h1 (128, 256)</div>
                <div class="ops-arrow">ReLU</div>
                <div class="ops-box hidden">r1 (128, 256)</div>
                <div class="ops-arrow">× W₂</div>
                <div class="ops-box output">lout (128, 10)</div>
              </div>
              <div class="ops-bars">
                <div v-for="(bar, idx) in [8, 12, 5, 10, 6, 78, 7, 16, 4, 12]" :key="idx" class="ops-bar-col">
                  <div class="ops-bar" :class="{ peak: idx === 5 }" :style="{ height: `${bar}%` }"></div>
                  <span>{{ idx }}</span>
                </div>
              </div>
            </div>

            <div v-else-if="currentSectionData.id === 'autograd'" class="viz-autograd">
              <div class="backward-chain">
                <div v-for="node in ['loss', 'mean_bwd', 'sum_bwd', 'mul_bwd', 'logsoftmax_bwd', 'matmul_bwd']" :key="node" class="back-node">
                  {{ node }}
                </div>
              </div>
              <div class="autograd-note">调度器只负责“往前驱递归”，局部梯度公式由每个 `*_backward` 自己实现。</div>
            </div>

            <div v-else-if="currentSectionData.id === 'training'" class="viz-training">
              <div class="training-loop">
                <div v-for="step in ['采样 batch', '前向传播', '计算 loss', 'backward()', 'SGD 更新', 'free 中间张量']" :key="step" class="loop-step">
                  {{ step }}
                </div>
              </div>
              <div class="training-meta">
                <span class="meta-pill">B = 128</span>
                <span class="meta-pill">lr = 0.005</span>
                <span class="meta-pill">steps = 20000</span>
              </div>
            </div>

            <div v-else-if="currentSectionData.id === 'performance'" class="viz-performance">
              <div class="perf-grid">
                <div class="perf-card">
                  <span class="perf-title">内存</span>
                  <span class="perf-value">64B aligned</span>
                </div>
                <div class="perf-card">
                  <span class="perf-title">前向 GEMM</span>
                  <span class="perf-value">vDSP_mmul</span>
                </div>
                <div class="perf-card">
                  <span class="perf-title">反向 GEMM</span>
                  <span class="perf-value">cblas_sgemm</span>
                </div>
                <div class="perf-card">
                  <span class="perf-title">初始化</span>
                  <span class="perf-value">Kaiming</span>
                </div>
              </div>
              <div class="memory-strip">
                <span class="live-tag keep">w1 / b1 / w2 / b2</span>
                <span class="live-tag temp">h1</span>
                <span class="live-tag temp">h1b</span>
                <span class="live-tag temp">r1</span>
                <span class="live-tag temp">...</span>
                <span class="live-tag free">free 每一步</span>
              </div>
            </div>
          </div>

          <div class="code-layout">
            <section class="code-panel">
              <div class="code-header">
                <div>
                  <div class="code-file">{{ currentSectionData.trainCode.file }}</div>
                  <div class="code-line">{{ currentSectionData.trainCode.lines }}</div>
                </div>
                <div class="code-caption">{{ currentSectionData.trainCode.title }}</div>
              </div>
              <pre class="code-block"><code>{{ currentSectionData.trainCode.code }}</code></pre>
            </section>

            <section class="code-panel secondary">
              <div class="code-header">
                <div>
                  <div class="code-file">{{ currentSectionData.tensorCode.file }}</div>
                  <div class="code-line">{{ currentSectionData.tensorCode.lines }}</div>
                </div>
                <div class="code-caption">{{ currentSectionData.tensorCode.title }}</div>
              </div>
              <pre class="code-block small"><code>{{ currentSectionData.tensorCode.code }}</code></pre>
            </section>
          </div>

          <div class="principles-grid">
            <article v-for="item in currentSectionData.bullets" :key="item.title" class="principle-card">
              <h3>{{ item.title }}</h3>
              <p>{{ item.text }}</p>
            </article>
          </div>
        </article>
      </main>
    </div>

    <section class="project-structure">
      <div class="section-kicker">文件地图</div>
      <h2>仓库里最值得先读的四个文件</h2>
      <div class="file-tree">
        <div class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">tensor.h</span>
          <span class="file-desc">张量结构、前向算子、backward 分发器、内存分配都在这里。</span>
        </div>
        <div class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">train.c</span>
          <span class="file-desc">CSV 读取、batch 采样、训练 loop、模型保存和耗时统计。</span>
        </div>
        <div class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">eval.c</span>
          <span class="file-desc">加载模型后跑测试集，验证训练结果到底有没有学会。</span>
        </div>
        <div class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">Makefile</span>
          <span class="file-desc">决定是走 Accelerate 还是 OpenMP，也是理解平台优化策略的入口。</span>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.learn-view {
  max-width: 1380px;
  margin: 0 auto;
  padding: 40px 24px 56px;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.page-header h1 {
  margin: 0 0 8px;
  font-size: 32px;
  font-weight: 800;
}

.page-desc {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-light);
}

.hero-panel {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
  gap: 24px;
  margin-bottom: 28px;
  padding: 28px;
  border-radius: 24px;
  background:
    radial-gradient(circle at top left, rgba(108, 99, 255, 0.16), transparent 38%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(255, 255, 255, 0.88));
  box-shadow: var(--shadow-md);
}

.section-kicker {
  margin-bottom: 10px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--color-primary);
}

.hero-copy h2 {
  margin: 0 0 12px;
  font-size: 28px;
  line-height: 1.2;
}

.hero-copy p {
  margin: 0;
  font-size: 15px;
  line-height: 1.75;
  color: var(--color-text-light);
}

.hero-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.metric-card {
  padding: 18px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid rgba(108, 99, 255, 0.08);
}

.metric-value {
  display: block;
  margin-bottom: 6px;
  font-size: 22px;
  font-weight: 800;
}

.metric-label {
  font-size: 12px;
  color: var(--color-text-light);
}

.learn-layout {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  gap: 24px;
  margin-bottom: 40px;
}

.section-nav {
  position: sticky;
  top: 88px;
  align-self: start;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.nav-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 14px 16px;
  border-radius: 14px;
  background: var(--color-card);
  color: var(--color-text-light);
  text-align: left;
  border: 1px solid rgba(108, 99, 255, 0.08);
}

.nav-item:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.nav-item.active {
  background: var(--color-primary);
  color: white;
}

.nav-label {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.nav-title {
  font-size: 14px;
  line-height: 1.4;
  font-weight: 700;
}

.content-card {
  padding: 30px;
  border-radius: 24px;
  background: var(--color-card);
  box-shadow: var(--shadow-md);
}

.content-head {
  margin-bottom: 24px;
}

.content-title {
  margin: 0 0 10px;
  font-size: 28px;
  line-height: 1.2;
}

.content-intro {
  margin: 0;
  font-size: 15px;
  line-height: 1.75;
  color: var(--color-text-light);
}

.visual-card {
  margin-bottom: 22px;
  padding: 24px;
  border-radius: 18px;
  background: var(--color-bg);
  box-shadow: inset 0 0 0 1px rgba(108, 99, 255, 0.06);
}

.file-lane,
.ops-flow,
.training-loop,
.backward-chain {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.file-box,
.ops-box,
.loop-step,
.back-node {
  padding: 14px 18px;
  border-radius: 14px;
  background: white;
  border: 2px solid var(--color-border);
  text-align: center;
}

.file-box.accent {
  border-color: var(--color-primary);
  background: rgba(108, 99, 255, 0.06);
}

.file-box.success {
  border-color: #22c55e;
  background: rgba(34, 197, 94, 0.07);
}

.file-name,
.panel-title,
.perf-title {
  display: block;
  font-size: 13px;
  font-weight: 800;
}

.file-role {
  display: block;
  margin-top: 4px;
  font-size: 11px;
  color: var(--color-text-light);
}

.file-arrow,
.ops-arrow {
  font-size: 22px;
  font-weight: 900;
  color: var(--color-primary);
}

.architecture-notes,
.training-meta,
.memory-strip,
.ops-bars {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 18px;
}

.note-chip,
.meta-pill,
.live-tag,
.autograd-note {
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(108, 99, 255, 0.08);
  font-size: 12px;
  color: var(--color-text-light);
}

.tensor-stack {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.tensor-panel {
  width: min(100%, 420px);
  padding: 18px;
  border-radius: 16px;
  background: white;
  border: 2px solid var(--color-border);
}

.tensor-panel.top {
  border-color: var(--color-primary);
}

.tensor-panel.bottom {
  border-color: #22c55e;
}

.tensor-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px dashed rgba(99, 110, 114, 0.18);
  font-size: 13px;
}

.tensor-row:last-child {
  border-bottom: none;
}

.stack-arrow {
  font-size: 24px;
  font-weight: 900;
  color: var(--color-primary);
}

.ops-box.input {
  border-color: var(--color-primary);
}

.ops-box.hidden {
  border-color: #f59e0b;
}

.ops-box.output {
  border-color: #22c55e;
}

.ops-bar-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--color-text-light);
}

.ops-bar {
  width: 22px;
  border-radius: 6px 6px 0 0;
  background: var(--color-primary);
  opacity: 0.5;
}

.ops-bar.peak {
  background: #22c55e;
  opacity: 1;
}

.viz-autograd {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.back-node {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.06);
  color: #ef4444;
  font-size: 12px;
  font-weight: 800;
}

.autograd-note {
  margin-top: 16px;
  border-radius: 12px;
  text-align: center;
}

.loop-step {
  border-color: rgba(108, 99, 255, 0.18);
  font-size: 13px;
  font-weight: 700;
}

.perf-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.perf-card {
  padding: 16px;
  border-radius: 16px;
  background: white;
  text-align: center;
  border: 1px solid rgba(108, 99, 255, 0.1);
}

.perf-value {
  display: block;
  margin-top: 6px;
  font-size: 18px;
  font-weight: 800;
}

.live-tag.keep {
  background: rgba(34, 197, 94, 0.1);
}

.live-tag.temp {
  background: rgba(245, 158, 11, 0.1);
}

.live-tag.free {
  background: rgba(239, 68, 68, 0.12);
}

.code-layout {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}

.code-panel {
  overflow: hidden;
  border-radius: 18px;
  background: #111827;
  box-shadow: var(--shadow-sm);
}

.code-panel.secondary {
  opacity: 0.92;
}

.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
}

.code-file {
  font-size: 13px;
  font-weight: 800;
  color: #f9fafb;
}

.code-line {
  margin-top: 2px;
  font-size: 11px;
  color: #9ca3af;
}

.code-caption {
  font-size: 11px;
  font-weight: 700;
  color: var(--color-primary);
}

.code-block {
  margin: 0;
  padding: 16px;
  overflow-x: auto;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12.5px;
  line-height: 1.7;
  color: #d1d5db;
  white-space: pre-wrap;
}

.code-block.small {
  font-size: 11.5px;
  color: #9ca3af;
}

.principles-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.principle-card {
  padding: 16px;
  border-radius: 14px;
  background: var(--color-bg);
  border-left: 3px solid var(--color-primary);
}

.principle-card h3 {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 800;
}

.principle-card p {
  margin: 0;
  font-size: 13px;
  line-height: 1.7;
  color: var(--color-text-light);
}

.project-structure {
  padding: 28px;
  border-radius: 24px;
  background: var(--color-card);
  box-shadow: var(--shadow-md);
}

.project-structure h2 {
  margin: 0 0 18px;
  font-size: 24px;
}

.file-tree {
  display: grid;
  gap: 10px;
}

.file-item {
  display: grid;
  grid-template-columns: 28px 120px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
  padding: 14px 16px;
  border-radius: 14px;
  background: var(--color-bg);
}

.file-icon {
  font-size: 18px;
}

.file-desc {
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text-light);
}

@media (max-width: 1100px) {
  .hero-panel,
  .learn-layout,
  .code-layout {
    grid-template-columns: 1fr;
  }

  .section-nav {
    position: static;
  }

  .perf-grid,
  .principles-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 768px) {
  .learn-view {
    padding: 28px 16px 40px;
  }

  .hero-copy h2,
  .content-title,
  .project-structure h2 {
    font-size: 24px;
  }

  .hero-metrics,
  .perf-grid,
  .principles-grid {
    grid-template-columns: 1fr;
  }

  .file-item {
    grid-template-columns: 28px 1fr;
  }

  .file-desc {
    grid-column: 2;
  }
}
</style>
