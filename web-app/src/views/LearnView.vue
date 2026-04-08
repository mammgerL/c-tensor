<script setup>
import { computed, ref } from 'vue'

const activeSection = ref('architecture')

const sections = [
  {
    id: 'architecture',
    label: '整体结构',
    title: '先认清这几个文件在分工什么',
    intro: '先别急着看细节。这个项目主要就是三个文件在配合：`tensor.h` 写怎么算，`train.c` 写怎么训练，`eval.c` 写怎么检查结果。',
    trainCode: {
      file: 'train.c',
      lines: '176-208',
      title: '先把输入、参数和训练配置准备好',
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
      title: '每种操作都有自己的编号',
      code: `#define MATMUL     0
#define MEAN       1
#define MUL        2
#define RELU       3
#define LOGSOFTMAX 4
#define SUM_AXIS1  5
#define ADD_BIAS   6`,
    },
    bullets: [
      { title: '模型先搭起来', text: '这里先把 `x`、`y`、`w1`、`b1`、`w2`、`b2` 都建好，后面训练就是一直在改这些值。' },
      { title: '`tensor.h` 最值得先读', text: '大部分“怎么算”的代码都在这里，所以先读懂这一份，后面会顺很多。' },
      { title: '训练和使用分开', text: '`train.c` 负责把模型训出来，网页和 `eval.c` 再把它读回来用。' },
    ],
  },
  {
    id: 'tensor',
    label: 'Tensor',
    title: 'Tensor 里不只放数值',
    intro: '`Arr` 负责把数字存进内存里，`Tensor` 再多记几件事：这个值是谁算出来的、梯度放哪、前面接着谁。',
    trainCode: {
      file: 'tensor.h',
      lines: '43-64',
      title: '先看 Arr 和 Tensor 里面装了什么',
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
      title: '创建时会顺手把形状和梯度也准备好',
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
      { title: '`data` 和 `grad` 要分开', text: '`data` 是当前算出来的值，`grad` 是等会儿反向传回来的改动方向。' },
      { title: '`prevs` 是回头找路用的', text: '反向传播时，程序就是顺着这里记下来的前驱一个个往回走。' },
      { title: '`strides` 是找位置用的', text: '虽然看起来像多维数组，但底层还是一块连续内存，`strides` 负责告诉程序该怎么跳。' },
    ],
  },
  {
    id: 'ops',
    label: '核心算子',
    title: '一张图片就是这样一路算下去的',
    intro: '图片先和 `W1` 做矩阵乘法，再过 ReLU，再和 `W2` 相乘，最后变成 10 个类别分数。每个算子算完都会把输入记下来，给后面的反向传播用。',
    trainCode: {
      file: 'tensor.h',
      lines: '415-451',
      title: '矩阵乘法把一层数据送到下一层',
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
      title: 'ReLU 和 LogSoftmax 负责把输出继续整理好',
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
      { title: '矩阵乘法最忙', text: '隐藏层和输出层都要做矩阵乘法，所以这里通常最花时间。' },
      { title: 'ReLU 很直白', text: '它把负数切成 0，正数原样留下，让网络能分出更复杂的数字区别。' },
      { title: 'LogSoftmax 会先稳一下', text: '它先减掉最大值再算指数，这样数字不会一下子冲得太大。' },
    ],
  },
  {
    id: 'autograd',
    label: '反向传播',
    title: '误差是怎么一路传回去的',
    intro: '`backward()` 本身不复杂。它先看当前这个 Tensor 是哪种操作算出来的，再叫对应的 `*_backward` 来处理，然后继续往前传。',
    trainCode: {
      file: 'tensor.h',
      lines: '211-220',
      title: '`backward()` 先决定该叫谁来算',
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
      title: '矩阵乘法会把梯度分回两边',
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
      { title: '每个算子只管自己的账', text: '比如 `relu_backward` 只处理 ReLU，`matmul_backward` 只处理矩阵乘法。' },
      { title: '梯度不是只来一次', text: '代码里用 `+=`，因为一个节点可能会从不同方向收到梯度。' },
      { title: '先从 loss 开始推', text: '训练时先把 `loss` 的梯度设成 `1.0f`，然后整条链才会开始往回传。' },
    ],
  },
  {
    id: 'training',
    label: '训练循环',
    title: '训练时每一轮都在重复什么',
    intro: '`train.c` 里的训练循环很直接：拿一个 batch，往前算，算出 loss，再往回传梯度，更新参数，最后把中间结果清掉。',
    trainCode: {
      file: 'train.c',
      lines: '228-286',
      title: '这一段就是一整轮训练',
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
      title: 'batch 是这样一批一批取出来的',
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
      { title: 'loss 也是拼出来的', text: '这里不是直接调一个现成大函数，而是按 `mul`、`sum_axis1`、`mean` 一步步算。' },
      { title: '样本会先打乱再取', text: '先把顺序洗一遍，再一段一段拿，这样每轮看到的数据更随机。' },
      { title: '训练时会顺手报进度', text: '程序会隔一段时间打印 loss 和耗时，方便看训练是不是在往前走。' },
    ],
  },
  {
    id: 'performance',
    label: '性能与内存',
    title: '想跑得稳又快，主要看这几件事',
    intro: '这里主要做三件事：把内存放整齐，矩阵乘法交给更快的库，每一步结束后把不用的 Tensor 及时释放。',
    trainCode: {
      file: 'tensor.h',
      lines: '88-125,163-167',
      title: '分配内存时就考虑后面怎么更快地读写',
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
      title: '初始化、更新参数和清理都在这里',
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
      { title: '内存摆整齐一点更快', text: '数据对齐后，底层库读起来更顺，矩阵乘法也更容易跑快。' },
      { title: '一开始怎么随机很重要', text: '权重如果初始得太大或太小，后面训练都可能不顺。' },
      { title: '中间结果不能一直留着', text: '像 `h1`、`r1` 这些只在这一轮有用，用完就要释放。' },
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
      <p class="page-desc">直接对着 `tensor.h` 和 `train.c` 看，这个纯 C 项目是怎么把模型训出来的。</p>
    </header>

    <section class="hero-panel">
      <div class="hero-copy">
        <div class="section-kicker">先看代码</div>
        <h2>代码写了什么，这里就讲什么</h2>
        <p>
          这页把训练过程拆成 6 小块。左边先用图把这一小块说清楚，右边再看真实代码，
          这样不用先背一堆词，也能知道每一段到底在干什么。
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
          <span class="metric-value">单头文件</span>
          <span class="metric-label">`tensor.h` 写法</span>
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
                  <span class="file-role">张量、算子、反向传播</span>
                </div>
                <div class="file-arrow">→</div>
                <div class="file-box accent">
                  <span class="file-name">train.c</span>
                  <span class="file-role">采样、训练、保存模型</span>
                </div>
                <div class="file-arrow">→</div>
                <div class="file-box success">
                  <span class="file-name">mnist_mlp.bin</span>
                  <span class="file-role">网页和 eval 都读它</span>
                </div>
              </div>
              <div class="architecture-notes">
                <div class="note-chip">训练代码先把模型文件写出来</div>
                <div class="note-chip">网页不会重新训练，只会把模型读进来</div>
                <div class="note-chip">大部分“怎么算”都写在 `tensor.h` 里</div>
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
              <div class="autograd-note">`backward()` 主要负责继续往前走，真正怎么算梯度，要看每个 `*_backward` 里的代码。</div>
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
      <h2>第一次看仓库，先读这四个文件</h2>
      <div class="file-tree">
        <div class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">tensor.h</span>
          <span class="file-desc">张量长什么样、前向怎么算、反向怎么传，基本都在这里。</span>
        </div>
        <div class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">train.c</span>
          <span class="file-desc">读数据、取 batch、训练、保存模型，这个文件从头串到尾。</span>
        </div>
        <div class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">eval.c</span>
          <span class="file-desc">把模型读回来，再跑测试集看看它到底学得怎么样。</span>
        </div>
        <div class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">Makefile</span>
          <span class="file-desc">这里能看到编译时走哪套加速方式，比如 Accelerate 或 OpenMP。</span>
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
