<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'

const currentStep = ref(0)

const trainingSteps = [
  {
    id: 'batch',
    label: '采样',
    title: '采样 Batch：从数据集取一批训练样本',
    desc: 'Fisher-Yates shuffle 从 60000 个训练样本中无放回地抽取 128 个，组成 mini-batch。用完所有样本后重新 shuffle，等价于按 epoch 训练。',
    trainCode: {
      file: 'train.c',
      lines: '232',
      code: `get_next_batch(&sampler,
    batch_x, batch_y,
    x, y, B);`,
    },
    tensorCode: {
      title: 'Fisher-Yates shuffle + memcpy',
      code: `// sampler_shuffle — Fisher-Yates
for (int i = s->N - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int tmp = s->idx[i];
    s->idx[i] = s->idx[j];
    s->idx[j] = tmp;
}

// get_next_batch — copy selected rows
for (int i = 0; i < B; i++) {
    int idx = s->idx[base + i];
    memcpy(bx + (size_t)i * D,
        X + (size_t)idx * D,
        (size_t)D * sizeof(float));
}`,
    },
    details: [
      { title: '无放回采样', text: '采样器把 60000 个样本打乱后顺序取 batch，用完再重洗。这样每个 epoch 内不会重复抽到同一个样本。' },
      { title: 'batch_y 的 -1', text: '标签是 one-hot 乘以 -1。后面做 mul(lout, batch_y) 时，正确类的对数概率会直接变成正的 NLL 项。' },
    ],
  },
  {
    id: 'hidden',
    label: '隐藏层',
    title: '前向传播：隐藏层 matmul → add_bias → relu',
    desc: '输入像素先映射到 256 维隐藏空间，再加偏置并经过 ReLU 截断负值。784 个像素会被压缩成可学习的特征。',
    trainCode: {
      file: 'train.c',
      lines: '235-237',
      code: `Tensor* h1 = matmul(batch_x, w1);   // (B,H)
Tensor* h1b = add_bias(h1, b1);    // (B,H)
Tensor* r1 = relu(h1b);            // (B,H)`,
    },
    tensorCode: {
      title: 'matmul (Accelerate / loops) + relu',
      code: `#if USE_ACCELERATE
vDSP_mmul(a->data->values, 1,
    b->data->values, 1,
    t->data->values, 1,
    (vDSP_Length)P, (vDSP_Length)R, (vDSP_Length)Q);
#else
for (int i = 0; i < P; i++) {
    for (int j = 0; j < R; j++) {
        float tmp = 0.0f;
        for (int k = 0; k < Q; k++) tmp += ...;
        t->data->values[i * R + j] = tmp;
    }
}
#endif

for (int i = 0; i < inp->data->size; i++)
    t->data->values[i] =
        (inp->data->values[i] > 0)
            ? inp->data->values[i] : 0.0f;`,
    },
    details: [
      { title: '为什么需要 ReLU', text: '如果没有非线性，两层线性层仍然可以合并成一层，模型表达能力不会提升。ReLU 让网络能学习更复杂的边界。' },
      { title: '维度变化', text: '(128, 784) × (784, 256) + (256) → (128, 256)。每个 batch 里的 128 张图都会得到 256 维隐藏表示。' },
    ],
  },
  {
    id: 'output',
    label: '输出层',
    title: '前向传播：输出层 matmul → add_bias → logsoftmax',
    desc: '隐藏特征投影到 10 个类别分数，再通过 LogSoftmax 变成数值稳定的对数概率分布。',
    trainCode: {
      file: 'train.c',
      lines: '239-242',
      code: `Tensor* h2 = matmul(r1, w2);        // (B,10)
Tensor* h2b = add_bias(h2, b2);      // (B,10)
Tensor* lout = logsoftmax(h2b);      // (B,10)`,
    },
    tensorCode: {
      title: 'logsoftmax — stable log probability',
      code: `float maxv = inp->data->values[base];
for (int c = 1; c < C; c++) {
    float v = inp->data->values[base + c * inp->data->strides[1]];
    if (v > maxv) maxv = v;
}

float sumexp = 0.0f;
for (int c = 0; c < C; c++) {
    float v = inp->data->values[base + c * inp->data->strides[1]];
    sumexp += expf(v - maxv);
}
float lse = logf(sumexp);

t->data->values[pos] =
    inp->data->values[pos] - maxv - lse;`,
    },
    details: [
      { title: '为什么是 LogSoftmax', text: '直接 softmax 再取 log 容易溢出。这里先减去每行最大值，再算 log(sum(exp))，数值更稳定。' },
      { title: '输出含义', text: '每个值都是 log(P(class))。越接近 0 代表概率越高，例如 log(0.9) 约等于 -0.105。' },
    ],
  },
  {
    id: 'loss',
    label: '损失',
    title: '计算损失：mul → sum_axis1 → mean',
    desc: '这里没有单独写一个 nll_loss，而是拆成三个基本 op。这样 autograd 可以给每个 op 单独定义 backward 逻辑。',
    trainCode: {
      file: 'train.c',
      lines: '244-246',
      code: `Tensor* mul_out = mul(lout, batch_y);      // (B,10)
Tensor* per_sample = sum_axis1(mul_out);   // (B,1)
Tensor* loss = mean(per_sample);           // (1)`,
    },
    tensorCode: {
      title: 'mul + sum_axis1 + mean',
      code: `for (int i = 0; i < a->data->size; i++)
    t->data->values[i] =
        a->data->values[i] * b->data->values[i];

for (int b = 0; b < B; b++) {
    float s = 0.0f;
    for (int c = 0; c < C; c++) s += ...;
    out->data->values[b] = s;
}

float s = 0.0f;
for (int i = 0; i < t->data->size; i++) s += t->data->values[i];
m->data->values[0] = s / (float)t->data->size;`,
    },
    details: [
      { title: '为什么拆成三步', text: 'autograd 的单位是单个 op。把损失写成 mul、sum_axis1、mean 三个原子操作，每个都能单独 backward。' },
      { title: '负号技巧', text: '因为 batch_y 的正确类是 -1，其余是 0，逐元素乘后按行求和，就得到了正确类的负对数概率。' },
    ],
  },
  {
    id: 'backward',
    label: '反向',
    title: '反向传播：backward() 递归回溯计算图',
    desc: 'loss 是标量，所以先手动放入种子梯度 1.0。然后 backward() 根据 op 类型分发到对应的 *_backward，再递归遍历 prevs。',
    trainCode: {
      file: 'train.c',
      lines: '249-250',
      code: `loss->grad->values[0] = 1.0f;
backward(loss);`,
    },
    tensorCode: {
      title: 'dispatch + recurse',
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
    details: [
      { title: '种子梯度', text: '标量 loss 满足 dloss/dloss = 1，所以 1.0f 是整条链式法则的起点。后面的所有梯度都是从这里递推出来的。' },
      { title: '递归调度', text: '计算图会从 mean 回到 sum_axis1，再回到 mul、logsoftmax、matmul，最终把梯度累加到 w1、b1、w2、b2。' },
    ],
  },
  {
    id: 'update',
    label: '更新',
    title: 'SGD 更新 + 手动清理中间张量',
    desc: '四组参数沿负梯度方向更新，随后把梯度归零并释放前向过程中产生的所有中间 Tensor。C 里这些都要手动完成。',
    trainCode: {
      file: 'train.c',
      lines: '256-286',
      code: `{
    float* w = w1->data->values;
    float* g = w1->grad->values;
    int sz = w1->data->size;
    for (int i = 0; i < sz; i++) {
        w[i] -= g[i] * lr;
        g[i] = 0.0f;
    }
}
// ... b1, w2, b2 同理

free_tensor(h1);
free_tensor(h1b);
free_tensor(r1);
free_tensor(h2);
free_tensor(h2b);
free_tensor(lout);
free_tensor(mul_out);
free_tensor(per_sample);
free_tensor(loss);`,
    },
    tensorCode: {
      title: 'free_tensor / free_arr',
      code: `static inline void free_tensor(Tensor* t) {
    if (!t) return;
    if (t->data) free_arr(t->data);
    if (t->grad) free_arr(t->grad);
    free(t);
}

static inline void free_arr(Arr* a) {
    if (!a) return;
    free(a->values);
    free(a->shape);
    free(a->strides);
    free(a);
}`,
    },
    details: [
      { title: '为什么要清零梯度', text: '这个 autograd 实现是累加梯度的。如果不把 g[i] 设回 0，下一步 backward 会把旧梯度一起叠上去。' },
      { title: '内存管理', text: '每一步都会生成 h1、h1b、r1、h2、h2b、lout、mul_out、per_sample、loss 等中间张量，不手动 free 就会持续泄漏。' },
    ],
  },
]

const currentStepData = computed(() => trainingSteps[currentStep.value])

function goToStep(index) {
  currentStep.value = index
}

function nextStep() {
  if (currentStep.value < trainingSteps.length - 1) currentStep.value++
}

function prevStep() {
  if (currentStep.value > 0) currentStep.value--
}

const simStep = ref(0)
const simLoss = ref(2.35)
const simAcc = ref(12.0)
const lossHistory = ref([])
const accHistory = ref([])
let simTimer = null

function simTick() {
  if (simStep.value >= 200) {
    stopSim()
    return
  }

  simStep.value++

  const targetLoss = 0.1 + 2.25 * Math.exp(-simStep.value / 30)
  simLoss.value = Math.max(0.05, targetLoss + (Math.random() - 0.5) * 0.05)

  const targetAcc = 96 - 84 * Math.exp(-simStep.value / 25)
  simAcc.value = Math.min(97, Math.max(10, targetAcc + (Math.random() - 0.5) * 0.5))

  lossHistory.value.push(simLoss.value)
  accHistory.value.push(simAcc.value)
}

function startSim() {
  stopSim()
  simStep.value = 0
  simLoss.value = 2.35
  simAcc.value = 12.0
  lossHistory.value = [2.35]
  accHistory.value = [12.0]
  simTimer = setInterval(simTick, 100)
}

function stopSim() {
  if (simTimer) {
    clearInterval(simTimer)
    simTimer = null
  }
}

function seededRandom(seed) {
  let s = seed
  return function () {
    s = (s * 16807) % 2147483647
    return (s - 1) / 2147483646
  }
}

const heatmapRows = 16
const heatmapCols = 16
const heatmapSlider = ref(0)

const heatmapState = computed(() => {
  if (heatmapSlider.value < 30) return 0
  if (heatmapSlider.value < 150) return 1
  return 2
})

function generateWeights(rows, cols, state) {
  const rng = seededRandom(42)
  const weights = []

  for (let r = 0; r < rows; r++) {
    const row = []
    for (let c = 0; c < cols; c++) {
      if (state === 0) {
        row.push((rng() - 0.5) * 2.0)
      } else if (state === 1) {
        const base = Math.sin(r * 0.3 + c * 0.2) * 0.5
        row.push(base + (rng() - 0.5) * 0.6)
      } else {
        const base = Math.sin(r * 0.3 + c * 0.2) * 0.8 + Math.cos(r * 0.1 - c * 0.4) * 0.3
        row.push(base + (rng() - 0.5) * 0.1)
      }
    }
    weights.push(row)
  }

  return weights
}

const heatmapWeights = computed(() => generateWeights(heatmapRows, heatmapCols, heatmapState.value))

function weightToColor(val) {
  const v = Math.max(-1, Math.min(1, val))
  if (v < 0) {
    const t = -v
    return `rgb(${Math.round(255 * (1 - t * 0.7))},${Math.round(255 * (1 - t * 0.7))},${Math.round(255 * (0.3 + 0.7 * t))})`
  }

  const t = v
  return `rgb(${Math.round(255 * (0.3 + 0.7 * t))},${Math.round(255 * (1 - t * 0.7))},${Math.round(255 * (1 - t * 0.7))})`
}

const heatmapStateLabel = computed(() => ['随机初始化', '训练中 (~5000 步)', '收敛 (~20000 步)'][heatmapState.value])

const lossCurvePath = computed(() => {
  if (lossHistory.value.length < 2) return ''

  const maxLoss = 2.5
  const width = 400
  const height = 150

  return lossHistory.value
    .map((value, index) => {
      const x = (index / 200) * width
      const y = height - (value / maxLoss) * height
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
})

const accCurvePath = computed(() => {
  if (accHistory.value.length < 2) return ''

  const width = 400
  const height = 150

  return accHistory.value
    .map((value, index) => {
      const x = (index / 200) * width
      const y = height - (value / 100) * height
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
})

const lossCurvePoint = computed(() => {
  if (lossHistory.value.length === 0) return null
  const index = lossHistory.value.length - 1
  const width = 400
  const height = 150
  return {
    x: (index / 200) * width,
    y: height - (lossHistory.value[index] / 2.5) * height,
  }
})

const accCurvePoint = computed(() => {
  if (accHistory.value.length === 0) return null
  const index = accHistory.value.length - 1
  const width = 400
  const height = 150
  return {
    x: (index / 200) * width,
    y: height - (accHistory.value[index] / 100) * height,
  }
})

onMounted(() => {
  startSim()
})

onUnmounted(() => {
  stopSim()
})
</script>

<template>
  <div class="training-view">
    <header class="page-header">
      <h1>训练过程详解</h1>
      <p class="page-desc">
        拆解 <code>train.c</code> 的核心训练循环，从采样到参数更新，每一步都对照真实 C 实现
      </p>
    </header>

    <section class="iteration-section">
      <div class="section-label">一次迭代详解</div>
      <p class="section-intro">把一次真实训练迭代拆成 6 个步骤。左边看数据和张量如何流动，右边直接对照仓库里的 C 代码实现。</p>

      <div class="step-bar">
        <button
          v-for="(step, idx) in trainingSteps"
          :key="step.id"
          :class="['step-tab', { active: currentStep === idx }]"
          @click="goToStep(idx)"
        >
          <span class="step-num">{{ idx + 1 }}</span>
          <span class="step-label">{{ step.label }}</span>
        </button>
      </div>

      <div class="iter-layout">
        <div class="viz-col">
          <div class="viz-card">
            <p class="step-tag">Step {{ currentStep + 1 }} · {{ currentStepData.trainCode.file }}:{{ currentStepData.trainCode.lines }}</p>
            <h2>{{ currentStepData.title }}</h2>
            <p class="viz-desc">{{ currentStepData.desc }}</p>

            <div class="visual-area">
              <div v-if="currentStepData.id === 'batch'" class="viz-batch">
                <div class="batch-flow">
                  <div class="dataset-grid-wrap">
                    <div class="dataset-grid">
                      <div
                        v-for="i in 24"
                        :key="i"
                        class="grid-cell"
                        :class="{ selected: [3, 5, 8, 9, 12, 15, 19, 22].includes(i) }"
                      ></div>
                      <div class="grid-label">训练集 (60000 × 784)</div>
                    </div>
                    <div class="viz-caption">紫色 = 当前 batch 选中的样本</div>
                  </div>
                  <div class="flow-arrow">→</div>
                  <div class="batch-result">
                    <div class="tensor-box blue">
                      <span class="tensor-name">batch_x</span>
                      <span class="tensor-dim">(128, 784)</span>
                      <span class="tensor-note">128 张 28×28 图片</span>
                    </div>
                    <div class="tensor-box pink">
                      <span class="tensor-name">batch_y</span>
                      <span class="tensor-dim">(128, 10)</span>
                      <span class="tensor-note">one-hot × (-1)</span>
                    </div>
                  </div>
                </div>
              </div>

              <div v-else-if="currentStepData.id === 'hidden'" class="viz-dims">
                <div class="dim-chain">
                  <div class="tensor-box blue">
                    <span class="tensor-name">batch_x</span>
                    <span class="tensor-dim">(128, 784)</span>
                  </div>
                  <span class="op-label">×</span>
                  <div class="tensor-box orange">
                    <span class="tensor-name">W₁</span>
                    <span class="tensor-dim">(784, 256)</span>
                  </div>
                  <span class="op-label">+</span>
                  <div class="tensor-box orange">
                    <span class="tensor-name">b₁</span>
                    <span class="tensor-dim">(256)</span>
                  </div>
                  <span class="op-label">→ ReLU →</span>
                  <div class="tensor-box green">
                    <span class="tensor-name">r1</span>
                    <span class="tensor-dim">(128, 256)</span>
                  </div>
                </div>
                <div class="relu-chart">
                  <svg viewBox="0 0 100 60" class="relu-svg">
                    <line x1="0" y1="30" x2="100" y2="30" stroke="var(--color-border)" stroke-width="1" />
                    <line x1="50" y1="0" x2="50" y2="60" stroke="var(--color-border)" stroke-width="1" />
                    <polyline points="10,30 50,30 90,2" fill="none" stroke="var(--color-primary)" stroke-width="2.5" stroke-linecap="round" />
                    <text x="92" y="8" font-size="8" fill="var(--color-primary)" font-weight="700">y</text>
                    <text x="52" y="58" font-size="7" fill="var(--color-text-light)">x</text>
                  </svg>
                  <span class="relu-label">ReLU: max(0, x)</span>
                </div>
                <div class="mini-note">
                  负值会被直接截成 0。没有这层非线性，两层 matmul 本质上仍然只是一层线性变换。
                </div>
              </div>

              <div v-else-if="currentStepData.id === 'output'" class="viz-dims">
                <div class="dim-chain">
                  <div class="tensor-box green">
                    <span class="tensor-name">r1</span>
                    <span class="tensor-dim">(128, 256)</span>
                  </div>
                  <span class="op-label">×</span>
                  <div class="tensor-box orange">
                    <span class="tensor-name">W₂</span>
                    <span class="tensor-dim">(256, 10)</span>
                  </div>
                  <span class="op-label">+</span>
                  <div class="tensor-box orange">
                    <span class="tensor-name">b₂</span>
                    <span class="tensor-dim">(10)</span>
                  </div>
                  <span class="op-label">→ LogSoftmax →</span>
                  <div class="tensor-box cyan">
                    <span class="tensor-name">lout</span>
                    <span class="tensor-dim">(128, 10)</span>
                  </div>
                </div>
                <div class="prob-chart">
                  <div v-for="(height, i) in [8, 12, 6, 10, 5, 80, 7, 18, 4, 14]" :key="i" class="prob-col">
                    <div class="prob-bar" :style="{ height: `${height}%` }" :class="{ correct: i === 5 }"></div>
                    <span class="prob-digit" :class="{ correct: i === 5 }">{{ i }}</span>
                  </div>
                </div>
                <div class="prob-note">LogSoftmax 输出：一个样本对 10 个数字的对数概率</div>
                <div class="mini-note center">这里示意里数字 5 的概率最高，所以它最可能成为当前样本的预测类别。</div>
              </div>

              <div v-else-if="currentStepData.id === 'loss'" class="viz-loss-flow">
                <div class="loss-chain">
                  <div class="loss-step">
                    <div class="tensor-box cyan">
                      <span class="tensor-name">lout × batch_y</span>
                      <span class="tensor-dim">(128,10)·(128,10)</span>
                    </div>
                    <div class="loss-step-label">逐元素乘</div>
                    <div class="loss-step-tip">正确类标签是 -1，其余是 0</div>
                  </div>
                  <div class="flow-arrow">→</div>
                  <div class="loss-step">
                    <div class="tensor-box orange">
                      <span class="tensor-name">sum_axis1</span>
                      <span class="tensor-dim">(128,10) → (128,1)</span>
                    </div>
                    <div class="loss-step-label">每行求和</div>
                    <div class="loss-step-tip">每个样本得到 1 个 NLL 值</div>
                  </div>
                  <div class="flow-arrow">→</div>
                  <div class="loss-step">
                    <div class="tensor-box red">
                      <span class="tensor-name">mean</span>
                      <span class="tensor-dim">(128,1) → (1)</span>
                    </div>
                    <div class="loss-step-label">batch 平均</div>
                    <div class="loss-step-tip">把 128 个样本压成 1 个标量</div>
                  </div>
                  <div class="flow-arrow">→</div>
                  <div class="loss-scalar">
                    <span class="scalar-label">loss</span>
                    <span class="scalar-value">{{ simLoss.toFixed(4) }}</span>
                  </div>
                </div>
              </div>

              <div v-else-if="currentStepData.id === 'backward'" class="viz-backward">
                <div class="bwd-chain">
                  <div
                    v-for="(node, i) in [
                      { name: 'loss', sub: 'grad=1.0' },
                      { name: 'mean_bwd', sub: '÷ 128' },
                      { name: 'sum_bwd', sub: 'broadcast' },
                      { name: 'mul_bwd', sub: '交叉乘' },
                      { name: 'logsoftmax_bwd', sub: 'J-softmax' },
                      { name: 'add_bias_bwd', sub: '累加' },
                      { name: 'matmul_bwd', sub: 'A^T·dC' },
                    ]"
                    :key="node.name"
                    class="bwd-node-group"
                  >
                    <div v-if="i > 0" class="bwd-arrow">←</div>
                    <div class="bwd-node" :style="{ opacity: 1 - i * 0.08 }">
                      <span class="bwd-name">{{ node.name }}</span>
                      <span class="bwd-sub">{{ node.sub }}</span>
                    </div>
                  </div>
                </div>
                <div class="bwd-code-hint">
                  <code>backward(t) { op_backward(t); for each prev: backward(prev); }</code>
                </div>
                <div class="mini-note center">每个节点先处理自己的局部梯度，再把结果递归分发给前驱节点。</div>
              </div>

              <div v-else-if="currentStepData.id === 'update'" class="viz-update">
                <div class="update-columns">
                  <div class="update-left">
                    <h4>SGD 参数更新</h4>
                    <div class="param-list">
                      <div
                        v-for="param in [
                          { name: 'W₁', count: '200704' },
                          { name: 'b₁', count: '256' },
                          { name: 'W₂', count: '2560' },
                          { name: 'b₂', count: '10' },
                        ]"
                        :key="param.name"
                        class="param-row"
                      >
                        <span class="param-name">{{ param.name }}</span>
                        <code class="param-formula">w -= g × 0.005; g = 0</code>
                        <span class="param-count">{{ param.count }} params</span>
                      </div>
                    </div>
                    <div class="param-total">总计 203,530 个参数</div>
                  </div>

                  <div class="update-right">
                    <h4>释放中间张量</h4>
                    <div class="free-list">
                      <div
                        v-for="tensor in ['h1 (128,256)', 'h1b (128,256)', 'r1 (128,256)', 'h2 (128,10)', 'h2b (128,10)', 'lout (128,10)', 'mul_out (128,10)', 'per_sample (128,1)', 'loss (1)']"
                        :key="tensor"
                        class="free-item"
                      >
                        {{ tensor }}
                      </div>
                    </div>
                    <div class="free-note">C 没有 GC，每一步都要手动 free</div>
                    <div class="free-note secondary">只有 w1、b1、w2、b2 和 batch 张量会保留到下一步</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="details-grid">
              <article v-for="detail in currentStepData.details" :key="detail.title" class="detail-item">
                <h3>{{ detail.title }}</h3>
                <p>{{ detail.text }}</p>
              </article>
            </div>
          </div>

          <div class="step-nav-bottom">
            <button class="nav-btn-prev" :disabled="currentStep === 0" @click="prevStep">← 上一步</button>
            <span class="nav-counter">{{ currentStep + 1 }} / {{ trainingSteps.length }}</span>
            <button class="nav-btn-next" :disabled="currentStep === trainingSteps.length - 1" @click="nextStep">下一步 →</button>
          </div>
        </div>

        <aside class="code-col">
          <div class="code-block">
            <div class="code-block-header">
              <div class="code-block-meta">
                <span class="code-block-title">train.c</span>
                <span class="code-block-subtitle">训练循环中的直接调用</span>
              </div>
              <span class="code-block-lines">:{{ currentStepData.trainCode.lines }}</span>
            </div>
            <pre class="code-content main-code"><code>{{ currentStepData.trainCode.code }}</code></pre>
          </div>

          <div class="code-block secondary">
            <div class="code-block-header">
              <div class="code-block-meta">
                <span class="code-block-title">tensor.h</span>
                <span class="code-block-subtitle">{{ currentStepData.tensorCode.title }}</span>
              </div>
              <span class="code-block-hint">核心实现</span>
            </div>
            <pre class="code-content tensor-code"><code>{{ currentStepData.tensorCode.code }}</code></pre>
          </div>
        </aside>
      </div>
    </section>

    <section class="progress-section">
      <div class="section-label">训练全过程</div>
      <p class="section-intro">上面解释一次 step 内部发生了什么，这里则从更长时间尺度看 loss、accuracy 和权重模式如何一起变化。</p>

      <div class="progress-layout">
        <div class="curve-card">
          <div class="curve-header">
            <h3>Loss & Accuracy</h3>
            <div class="curve-stats">
              <span class="stat loss">Loss: {{ simLoss.toFixed(4) }}</span>
              <span class="stat acc">Acc: {{ simAcc.toFixed(1) }}%</span>
              <span class="stat step">Step: {{ simStep * 100 }} / 20000</span>
            </div>
          </div>
          <p class="curve-subtitle">模拟 20000 个训练 step 的宏观走势，loss 指数下降，accuracy 逐步收敛到高位。</p>

          <svg viewBox="0 0 400 150" class="curve-svg" preserveAspectRatio="none">
            <path d="M0,150 L400,150" class="curve-axis"></path>
            <path d="M0,0 L0,150" class="curve-axis"></path>
            <path :d="lossCurvePath" fill="none" stroke="var(--color-primary)" stroke-width="2.5" opacity="0.85" />
            <path :d="accCurvePath" fill="none" stroke="var(--color-success)" stroke-width="2.5" opacity="0.85" />
            <circle v-if="lossCurvePoint" :cx="lossCurvePoint.x" :cy="lossCurvePoint.y" r="4.5" fill="var(--color-primary)" />
            <circle v-if="accCurvePoint" :cx="accCurvePoint.x" :cy="accCurvePoint.y" r="4.5" fill="var(--color-success)" />
          </svg>
          <div class="curve-footer">
            <span>Step 0</span>
            <span>Step 10000</span>
            <span>Step 20000</span>
          </div>

          <div class="curve-legend">
            <span class="legend-item"><span class="legend-dot loss"></span>Loss 下降</span>
            <span class="legend-item"><span class="legend-dot acc"></span>Accuracy 上升</span>
          </div>
        </div>

        <div class="heatmap-card">
          <div class="heatmap-header">
            <h3>权重演变 (W₁ 局部)</h3>
            <span class="heatmap-state">{{ heatmapStateLabel }}</span>
          </div>
          <p class="curve-subtitle">拖动滑块观察从随机初始化到形成结构，再到基本收敛时的局部权重模式。</p>
          <div class="heatmap-phases">
            <span :class="['phase-pill', { active: heatmapState === 0 }]">随机</span>
            <span :class="['phase-pill', { active: heatmapState === 1 }]">学习中</span>
            <span :class="['phase-pill', { active: heatmapState === 2 }]">收敛</span>
          </div>

          <div class="heatmap-grid">
            <div v-for="(row, ri) in heatmapWeights" :key="ri" class="heatmap-row">
              <div
                v-for="(val, ci) in row"
                :key="ci"
                class="heatmap-cell"
                :style="{ background: weightToColor(val) }"
                :title="`W[${ri}][${ci}] = ${val.toFixed(3)}`"
              ></div>
            </div>
          </div>

          <div class="heatmap-controls">
            <input v-model.number="heatmapSlider" type="range" min="0" max="200" class="heatmap-range" />
            <div class="heatmap-ticks">
              <span>Step 0</span>
              <span>Step 10000</span>
              <span>Step 20000</span>
            </div>
          </div>

          <div class="heatmap-scale">
            <span>-1.0</span>
            <div class="scale-bar">
              <div class="scale-neg"></div>
              <div class="scale-zero"></div>
              <div class="scale-pos"></div>
            </div>
            <span>+1.0</span>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.training-view {
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 24px 56px;
}

.page-header {
  text-align: center;
  margin-bottom: 40px;
}

.page-header h1 {
  margin: 0 0 10px;
  font-size: 32px;
  font-weight: 800;
}

.page-desc {
  margin: 0;
  font-size: 16px;
  color: var(--color-text-light);
}

.page-desc code {
  background: var(--color-border);
  padding: 2px 7px;
  border-radius: 5px;
  font-size: 14px;
}

.iteration-section {
  margin-bottom: 8px;
}

.section-label {
  margin-bottom: 16px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--color-primary);
}

.section-intro {
  max-width: 760px;
  margin: -6px 0 22px;
  font-size: 14px;
  line-height: 1.7;
  color: var(--color-text-light);
}

.step-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 24px;
}

.step-tab {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border-radius: 12px;
  border: 1px solid var(--color-border);
  background: var(--color-card);
  color: var(--color-text-light);
  font-size: 13px;
  font-weight: 700;
  transition: all 0.2s ease;
}

.step-tab:hover {
  border-color: var(--color-primary);
  transform: translateY(-1px);
}

.step-tab.active {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

.step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.2);
  font-size: 11px;
}

.step-label {
  white-space: nowrap;
}

.iter-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 400px;
  gap: 24px;
}

.viz-col {
  min-width: 0;
}

.viz-card {
  padding: 28px;
  border-radius: 20px;
  background: var(--color-card);
  box-shadow: var(--shadow-md);
  border: 1px solid rgba(108, 99, 255, 0.08);
}

.step-tag {
  margin: 0 0 6px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--color-primary);
}

.viz-card h2 {
  margin: 0 0 10px;
  font-size: 22px;
  font-weight: 800;
}

.viz-desc {
  margin: 0 0 24px;
  font-size: 15px;
  line-height: 1.7;
  color: var(--color-text-light);
}

.visual-area {
  min-height: 320px;
  padding: 28px;
  margin-bottom: 20px;
  border-radius: 16px;
  background: var(--color-bg);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 0 0 1px rgba(108, 99, 255, 0.06);
}

.viz-caption {
  margin-top: 8px;
  font-size: 10px;
  font-weight: 700;
  color: #9ca3af;
}

.mini-note {
  max-width: 420px;
  margin: 18px auto 0;
  padding: 10px 14px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.62);
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-text-light);
  text-align: left;
}

.mini-note.center {
  text-align: center;
}

.details-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.detail-item {
  padding: 14px;
  border-left: 3px solid var(--color-primary);
  border-radius: 10px;
  background: var(--color-bg);
}

.detail-item h3 {
  margin: 0 0 6px;
  font-size: 14px;
  font-weight: 700;
}

.detail-item p {
  margin: 0;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text-light);
}

.step-nav-bottom {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-top: 20px;
}

.nav-btn-prev,
.nav-btn-next {
  padding: 10px 20px;
  border-radius: 10px;
  border: 1px solid var(--color-border);
  background: var(--color-card);
  color: var(--color-text);
  font-size: 13px;
  font-weight: 700;
}

.nav-btn-prev:hover:not(:disabled),
.nav-btn-next:hover:not(:disabled) {
  transform: translateY(-1px);
}

.nav-btn-next {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

.nav-btn-prev:disabled,
.nav-btn-next:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.nav-counter {
  font-size: 13px;
  font-weight: 700;
  color: var(--color-text-light);
}

.code-col {
  position: sticky;
  top: 24px;
  align-self: start;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.code-block {
  overflow: hidden;
  border-radius: 16px;
  background: #111827;
  box-shadow: var(--shadow-md);
}

.code-block.secondary {
  opacity: 0.9;
}

.code-block-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.05);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.code-block-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.code-block-title {
  font-size: 13px;
  font-weight: 800;
  color: #e5e7eb;
}

.code-block-subtitle {
  font-size: 10px;
  font-weight: 700;
  color: #6b7280;
}

.code-block-lines {
  font-size: 12px;
  font-weight: 700;
  color: var(--color-primary);
  font-family: 'SF Mono', Monaco, monospace;
}

.code-block-hint {
  font-size: 11px;
  font-weight: 700;
  color: #9ca3af;
}

.code-content {
  margin: 0;
  padding: 16px;
  overflow-x: auto;
  font-family: 'SF Mono', Monaco, monospace;
  line-height: 1.65;
  white-space: pre-wrap;
}

.main-code code {
  font-size: 13px;
  color: #e5e7eb;
}

.tensor-code code {
  font-size: 11.5px;
  color: #9ca3af;
}

.viz-batch {
  text-align: center;
}

.batch-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  flex-wrap: wrap;
}

.dataset-grid-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.dataset-grid {
  display: grid;
  grid-template-columns: repeat(6, 20px);
  gap: 4px;
  position: relative;
  padding-bottom: 24px;
}

.grid-cell {
  width: 20px;
  height: 20px;
  border-radius: 4px;
  background: var(--color-border);
  transition: background 0.3s;
}

.grid-cell.selected {
  background: var(--color-primary);
  opacity: 0.8;
}

.grid-label {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  font-size: 10px;
  font-weight: 700;
  text-align: center;
  color: var(--color-text-light);
}

.flow-arrow {
  font-size: 28px;
  font-weight: 900;
  color: var(--color-primary);
}

.batch-result {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.tensor-box {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 16px;
  border: 2px solid;
  border-radius: 10px;
  text-align: center;
}

.tensor-box.blue {
  border-color: var(--color-primary);
  background: rgba(108, 99, 255, 0.05);
}

.tensor-box.pink {
  border-color: var(--color-secondary);
  background: rgba(255, 107, 157, 0.05);
}

.tensor-box.orange {
  border-color: #f59e0b;
  background: rgba(245, 158, 11, 0.05);
}

.tensor-box.green {
  border-color: #22c55e;
  background: rgba(34, 197, 94, 0.05);
}

.tensor-box.cyan {
  border-color: var(--color-accent);
  background: rgba(0, 210, 255, 0.05);
}

.tensor-box.red {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.05);
}

.tensor-name {
  font-size: 12px;
  font-weight: 800;
}

.tensor-dim {
  font-size: 15px;
  font-weight: 800;
}

.tensor-note {
  font-size: 10px;
  color: var(--color-text-light);
}

.viz-dims {
  width: 100%;
  text-align: center;
}

.dim-chain {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 20px;
}

.op-label {
  font-size: 14px;
  font-weight: 800;
  color: var(--color-text-light);
}

.relu-chart {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin-top: 8px;
}

.relu-svg {
  width: 80px;
  height: 50px;
}

.relu-label {
  font-size: 12px;
  font-weight: 700;
  color: var(--color-text-light);
}

.prob-chart {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 6px;
  height: 100px;
  margin-top: 16px;
}

.prob-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.prob-bar {
  width: 28px;
  border-radius: 4px 4px 0 0;
  background: var(--color-primary);
  opacity: 0.5;
  transition: height 0.3s;
}

.prob-bar.correct {
  background: var(--color-success);
  opacity: 1;
}

.prob-digit {
  font-size: 11px;
  font-weight: 700;
  color: var(--color-text-light);
}

.prob-digit.correct {
  font-weight: 800;
  color: var(--color-success);
}

.prob-note {
  margin-top: 10px;
  font-size: 12px;
  text-align: center;
  color: var(--color-text-light);
}

.viz-loss-flow {
  width: 100%;
  text-align: center;
}

.loss-chain {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
}

.loss-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.loss-step-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--color-text-light);
}

.loss-step-tip {
  max-width: 140px;
  font-size: 10px;
  line-height: 1.45;
  color: #9ca3af;
}

.loss-scalar {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 14px 22px;
  border-radius: 12px;
  background: #ef4444;
  color: white;
}

.scalar-label {
  font-size: 11px;
  font-weight: 700;
}

.scalar-value {
  font-size: 22px;
  font-weight: 800;
}

.viz-backward {
  width: 100%;
  text-align: center;
}

.bwd-chain {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  flex-wrap: wrap;
}

.bwd-node-group {
  display: flex;
  align-items: center;
  gap: 4px;
}

.bwd-arrow {
  font-size: 16px;
  font-weight: 900;
  color: #ef4444;
}

.bwd-node {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 12px;
  border: 2px solid #ef4444;
  border-radius: 8px;
  background: rgba(239, 68, 68, 0.06);
  text-align: center;
}

.bwd-name {
  font-size: 11px;
  font-weight: 800;
  color: #ef4444;
}

.bwd-sub {
  font-size: 9px;
  color: var(--color-text-light);
}

.bwd-code-hint {
  display: inline-block;
  margin-top: 16px;
  padding: 10px 16px;
  border-radius: 8px;
  background: rgba(239, 68, 68, 0.06);
}

.bwd-code-hint code {
  font-size: 12px;
  color: #ef4444;
}

.viz-update {
  width: 100%;
}

.update-columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  text-align: left;
}

.update-columns h4 {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 800;
}

.param-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.param-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  background: rgba(108, 99, 255, 0.06);
  font-size: 12px;
}

.param-name {
  min-width: 24px;
  font-weight: 800;
  color: #f59e0b;
}

.param-formula {
  flex: 1;
  font-size: 11px;
  font-family: 'SF Mono', Monaco, monospace;
}

.param-count {
  white-space: nowrap;
  font-size: 10px;
  color: var(--color-text-light);
}

.param-total {
  margin-top: 8px;
  font-size: 11px;
  color: var(--color-text-light);
}

.free-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.free-item {
  padding: 5px 10px;
  border-radius: 6px;
  background: rgba(239, 68, 68, 0.06);
  font-size: 11px;
  font-family: 'SF Mono', Monaco, monospace;
  color: #ef4444;
  text-decoration: line-through;
  opacity: 0.65;
}

.free-note {
  margin-top: 8px;
  font-size: 11px;
  color: var(--color-text-light);
}

.free-note.secondary {
  margin-top: 4px;
}

.progress-section {
  margin-top: 48px;
  padding-top: 40px;
  border-top: 2px solid var(--color-border);
}

.progress-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.curve-card,
.heatmap-card {
  padding: 24px;
  border-radius: 20px;
  background: var(--color-card);
  box-shadow: var(--shadow-md);
  border: 1px solid rgba(108, 99, 255, 0.08);
}

.curve-header,
.heatmap-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.curve-header h3,
.heatmap-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 800;
}

.curve-stats {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.stat {
  padding: 4px 8px;
  border-radius: 6px;
  background: var(--color-bg);
  font-size: 12px;
  font-weight: 700;
}

.stat.loss {
  color: var(--color-primary);
}

.stat.acc {
  color: var(--color-success);
}

.stat.step {
  color: var(--color-text-light);
}

.curve-subtitle {
  margin: -2px 0 14px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-text-light);
}

.curve-svg {
  width: 100%;
  height: 150px;
  border-radius: 10px;
  background: var(--color-bg);
  background-image:
    linear-gradient(to right, rgba(99, 110, 114, 0.08) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(99, 110, 114, 0.08) 1px, transparent 1px);
  background-size: 20% 100%, 100% 25%;
}

.curve-axis {
  fill: none;
  stroke: rgba(99, 110, 114, 0.15);
  stroke-width: 1;
}

.curve-legend {
  display: flex;
  gap: 20px;
  margin-top: 10px;
  font-size: 12px;
  font-weight: 700;
}

.curve-footer {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 10px;
  color: var(--color-text-light);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--color-text-light);
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.legend-dot.loss {
  background: var(--color-primary);
}

.legend-dot.acc {
  background: var(--color-success);
}

.heatmap-state {
  font-size: 12px;
  font-weight: 700;
  color: var(--color-primary);
}

.heatmap-phases {
  display: flex;
  gap: 8px;
  margin: -2px 0 14px;
  flex-wrap: wrap;
}

.phase-pill {
  padding: 5px 10px;
  border-radius: 999px;
  background: var(--color-bg);
  color: var(--color-text-light);
  font-size: 11px;
  font-weight: 800;
  transition: all 0.2s ease;
}

.phase-pill.active {
  background: var(--color-primary);
  color: white;
}

.heatmap-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
  align-items: center;
  margin-bottom: 16px;
}

.heatmap-row {
  display: flex;
  gap: 2px;
}

.heatmap-cell {
  width: 18px;
  height: 18px;
  border-radius: 3px;
  transition: background 0.4s ease, transform 0.15s ease;
  cursor: pointer;
}

.heatmap-cell:hover {
  position: relative;
  z-index: 1;
  transform: scale(1.3);
}

.heatmap-controls {
  margin-bottom: 12px;
}

.heatmap-range {
  width: 100%;
  accent-color: var(--color-primary);
}

.heatmap-ticks {
  display: flex;
  justify-content: space-between;
  margin-top: 4px;
  font-size: 10px;
  color: var(--color-text-light);
}

.heatmap-scale {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 11px;
  font-weight: 700;
  color: var(--color-text-light);
}

.scale-bar {
  display: flex;
  width: 120px;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
}

.scale-neg {
  flex: 1;
  background: rgb(76, 76, 204);
}

.scale-zero {
  flex: 1;
  background: rgb(200, 200, 200);
}

.scale-pos {
  flex: 1;
  background: rgb(204, 76, 76);
}

@media (max-width: 1100px) {
  .iter-layout {
    grid-template-columns: 1fr;
  }

  .code-col {
    position: static;
  }

  .progress-layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .training-view {
    padding: 28px 16px 40px;
  }

  .page-header h1 {
    font-size: 28px;
  }

  .section-intro {
    font-size: 13px;
  }

  .visual-area {
    min-height: 260px;
    padding: 18px;
  }

  .details-grid {
    grid-template-columns: 1fr;
  }

  .step-nav-bottom {
    flex-wrap: wrap;
    justify-content: center;
  }

  .update-columns {
    grid-template-columns: 1fr;
  }

  .curve-legend {
    flex-wrap: wrap;
    gap: 12px;
  }

  .curve-header,
  .heatmap-header {
    align-items: flex-start;
  }
}
</style>
