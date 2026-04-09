<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

// ── Network architecture diagram (3b1b-style) ──────────────────────
// Fake activations simulating recognition of a "7"
const diagramLayers = [
  { x: 80,  label: '输入层', sub: '784 个像素', color: '#6C63FF',
    act: [0, 0, .15, .9, .85, .4, .05, 0] },
  { x: 300, label: '隐藏层', sub: '256 个特征', color: '#FF6B9D',
    act: [.7, 0, .4, .9, 0, .2, .6, .1] },
  { x: 520, label: '输出层', sub: '10 个数字', color: '#00D2FF',
    act: [.02, .01, .05, .02, .01, .01, .01, .88, .01, .02] },
]

const diagramNodes = computed(() =>
  diagramLayers.map(l => ({
    ...l,
    nodes: l.act.map((a, i) => ({
      y: (260 / (l.act.length + 1)) * (i + 1) + 15,
      a,
    })),
  }))
)

const diagramEdges = computed(() => {
  const edges = []
  const layers = diagramNodes.value
  for (let l = 0; l < layers.length - 1; l++) {
    const from = layers[l], to = layers[l + 1]
    for (const fn of from.nodes) {
      for (const tn of to.nodes) {
        edges.push({
          x1: from.x, y1: fn.y, x2: to.x, y2: tn.y,
          opacity: 0.04 + fn.a * tn.a * 0.25,
        })
      }
    }
  }
  return edges
})

// Weight pattern grid (what a hidden neuron "looks for" — looks like a loop)
const weightGrid = [
  [-.1,  0,  .2,  .5,  .3,  0, -.1],
  [-.1,  .1, .6,  .9,  .7,  .2, -.1],
  [ 0,   .3, .9,  .15, .1,  .8,  .1],
  [ 0,   .6, .3, -.1,   0,  .5,  .2],
  [ .1,  .7, .1,   0,  .2,  .7,  .1],
  [-.1,  .2, .5,  .7,  .8,  .3,   0],
  [-.1,  0,  .1,  .3,  .2,   0, -.1],
]

// ── Matmul worked example (4×3 for illustration) ────────────────────
// Real network: x [1,784] × W1 [784,256] = h [1,256]
// Toy example:  x [1,4]   × W  [4,3]     = h [1,3]
const matmulX = [0.8, -1.0, 0.5, -0.3]
const matmulW = [
  [ 0.2, -0.1,  0.4],
  [ 0.3,  0.5, -0.2],
  [-0.1,  0.3,  0.1],
  [ 0.4, -0.2,  0.6],
]
// h[j] = Σ x[i] * W[i][j]
const matmulH = matmulW[0].map((_, j) => {
  const terms = matmulX.map((xi, i) => xi * matmulW[i][j])
  return { terms, sum: terms.reduce((a, b) => a + b, 0) }
})
const matmulHighlight = ref(0) // which output column to highlight

// Bias + ReLU continuation of the matmul example
const biasB = [0.10, -0.20, 0.05]
const afterBias = matmulH.map((h, j) => h.sum + biasB[j])
const afterRelu = afterBias.map(v => Math.max(0, v))

// Softmax output bars for a "7"
const softmaxBars = [
  { digit: 0, prob: 0.002 },
  { digit: 1, prob: 0.004 },
  { digit: 2, prob: 0.045 },
  { digit: 3, prob: 0.008 },
  { digit: 4, prob: 0.001 },
  { digit: 5, prob: 0.018 },
  { digit: 6, prob: 0.001 },
  { digit: 7, prob: 0.881 },
  { digit: 8, prob: 0.003 },
  { digit: 9, prob: 0.040 },
]

const features = [
  {
    icon: '📐',
    title: '矩阵计算演示',
    desc: '手写一个数字，逐步查看 784→256→10 的矩阵乘法、ReLU、LogSoftmax 计算过程',
    path: '/playground',
    color: '#6C63FF',
  },
  {
    icon: '🚀',
    title: '训练过程',
    desc: '把一次训练迭代拆成采样、两段前向传播、损失、反向传播和参数更新 6 个步骤来看。',
    path: '/training',
    color: '#22C55E',
  },
  {
    icon: '⚙️',
    title: 'C 代码原理',
    desc: '了解 tensor.h 的核心数据结构、自动微分、Kaiming 初始化等实现细节',
    path: '/learn',
    color: '#FF6B9D',
  },
  {
    icon: '🔍',
    title: '数据探索',
    desc: '浏览 MNIST 测试集，分析哪些数字容易被误识别，理解模型边界情况',
    path: '/explore',
    color: '#00D2FF',
  },
]

const intuitions = [
  {
    num: '01',
    title: '一个"神经元"就是一个数字',
    body: '输入层的 784 个神经元，是 28×28 每个像素归一化后的值 (−1 ~ 1，−1 = 黑色背景，+1 = 白色笔画)。输出层的 10 个神经元，分别代表"这是 0 的信心、是 1 的信心…是 9 的信心"。隐藏层的 256 个神经元呢？—— 那正是网络要"学出来"的东西。',
    visual: 'neurons',
  },
  {
    num: '02',
    title: '每个神经元在"寻找"一种模式',
    body: '一个隐藏层神经元接收全部 784 个输入，每个输入配一个权重 w。正权重表示"这个位置越亮越好"，负权重表示"这个位置越暗越好"，接近 0 表示"这个位置无所谓"。把 784 个权重排回 28×28，就是这个神经元在寻找的模式。',
    visual: 'weights',
  },
  {
    num: '03',
    title: '矩阵乘法 = 256 个神经元同时思考',
    body: 'matmul(x, W1) 不是什么数学把戏 —— 它就是 256 个隐藏层神经元同时对同一张图做加权和。W1 的每一列就是一个神经元的权重向量。深度学习离不开矩阵，是因为矩阵就是"并行的思考"。',
    visual: 'formula',
    formulaText: 'x [1,784]  ×  W1 [784,256]  =  h [1,256]',
  },
  {
    num: '04',
    title: '整个网络只是 203,530 个旋钮',
    body: '784×256 (W1) + 256 (b1) + 256×10 (W2) + 10 (b2) = 203,530 个参数。训练就是在这 203,530 维空间里，找到一组让模型答对的数字。"梯度下降"就是在问：每个旋钮该往哪拧？拧多少？',
    visual: 'params',
  },
]

const principles = [
  {
    num: '01',
    title: '归一化像素',
    shape: '[1, 784]',
    why: '像素值从 0 到 255 差距太大，直接喂进网络会让前几步的加权和数值爆炸、梯度也不稳定。',
    how: '除以 127.5 再减 1，把每个像素压到 [−1, 1]。黑色背景 (0) 变成 −1，白色笔画 (255) 变成 +1，灰色抗锯齿像素落在中间。',
    code: 'x[i] = pixel / 127.5 - 1.0',
    example: `一个亮度 230 的灰色像素：
x = 230 / 127.5 − 1 = 0.804

黑色背景像素 (0) 会变成 −1，纯白笔画 (255) 会变成 +1。`,
  },
  {
    num: '02',
    title: '第一次矩阵乘法 · 256 个模式探测器同时工作',
    shape: '[1, 784] → [1, 256]',
    why: '784 个原始像素太低级了，我们需要从里面抽出"更有意义"的特征 —— 边缘、弧线、笔画方向…',
    how: '256 个神经元各自对 784 个像素做一次加权和，总共 784 × 256 = 200,704 次乘加。W1 的每一列，就是一个神经元想找的那张图。',
    code: 'h1 = matmul(x, W1)   // x @ W1',
    example: `拿第 0 号隐藏神经元来算：
h1[0] = x[0]·W1[0,0] + x[1]·W1[1,0] + ... + x[783]·W1[783,0]
      = (−1.0)·(−0.04) + (−1.0)·(0.02) + ... + 0.804·0.12 + ...
      = 2.05

784 项相加就是这一个数。256 个神经元一起做，就得到一个 256 维向量。`,
  },
  {
    num: '03',
    title: '加偏置 · 每个神经元的激活门槛',
    shape: '[1, 256]',
    why: '有些模式只要稍微像就该触发，有些要非常像才算数 —— 神经元之间的"挑剔程度"不该是一样的。',
    how: '给 256 个神经元各加一个可学习的 b。b 越负，这个神经元越挑剔；越正，越容易被点亮。',
    code: 'h1 = h1 + b1',
    example: `继续上面那个神经元：
h1[0] = 2.05 + b1[0]
      = 2.05 + 0.08
      = 2.13

如果 b1[0] = −5，这个神经元哪怕看到 2.05 的匹配度也会被压到负数，下一步 ReLU 就会把它关掉。`,
  },
  {
    num: '04',
    title: 'ReLU · 没看到就闭嘴',
    shape: '[1, 256]',
    why: '如果每一层都是线性的，堆再多层也只是一次线性变换。网络必须有"非线性决策"的能力。',
    how: '负值归零。就这么一个简单操作，让每个神经元能自由地"开/关"，256 个神经元组合出指数级的决策空间。',
    code: 'h1 = max(0, h1)',
    example: `假设前 5 个神经元加完偏置后是：
h1 = [ 2.13, −0.47,  0.88, −1.20,  3.02, ... ]
   → [ 2.13,    0  ,  0.88,    0  ,  3.02, ... ]

第 1、3 号神经元"关闭"（归零），剩下的保持不变。256 个里通常有一半左右会被 ReLU 关掉。`,
  },
  {
    num: '05',
    title: '第二次矩阵乘法 · 把特征翻译成数字',
    shape: '[1, 256] → [1, 10]',
    why: '现在我们有 256 个"抽象特征"，但用户要的是 0~9 中的一个答案。需要把特征再组合一次。',
    how: '10 个输出神经元，每个对 256 个特征做加权和（再加 b2）。W2 的第 c 列告诉我们"数字 c 喜欢哪些特征"。',
    code: 'h2 = matmul(h1, W2) + b2',
    example: `对"是不是 7"的那个输出（索引 7）：
h2[7] = h1[0]·W2[0,7] + h1[1]·W2[1,7] + ... + h1[255]·W2[255,7] + b2[7]
      = 2.13·0.41 + 0·(−0.22) + 0.88·0.67 + ... + 0.08
      = 5.12

10 个类别同时算完，得到 10 个原始分数（logits）：
h2 = [−1.02, −0.38, 2.15, 0.47, −2.33, 1.20, −1.85,  5.12, −0.76, 2.04]
       0      1     2     3     4      5     6       7      8     9
"7"的分数最高，其它类别被压得很低。`,
  },
  {
    num: '06',
    title: 'LogSoftmax · 把原始分数变成概率',
    shape: '[1, 10]',
    why: '10 个原始分数不能直接当信心用。直接 exp/sum 又会数值溢出（exp(100) 就爆了）。',
    how: '先减去最大值再做 log-sum-exp：数值稳定，并且和训练时的 NLL loss 天然配合 (log-prob 相加即可)。',
    code: 'out[c] = h2[c] − max − log(Σ exp(h2 − max))',
    example: `h2 = [−1.02, −0.38, 2.15, 0.47, −2.33, 1.20, −1.85, 5.12, −0.76, 2.04]
max = 5.12

1. 每个值减去 max：
   centered = [−6.14, −5.50, −2.97, ..., 0.00, ..., −3.08]
2. 算 log(Σ exp(centered))：
   Σ exp = 1.00 + 0.0513 + 0.00957 + ... ≈ 1.135
   log(1.135) = 0.127
3. 每个值都减这个 log-sum-exp：
   out[7] = 5.12 − 5.12 − 0.127 = −0.127

out[c] 是对数概率，exp(out[c]) 是真正的概率，所有类别概率加起来 = 1。`,
  },
  {
    num: '07',
    title: 'Argmax · 模型的最终答案',
    shape: '[1, 10] → 1',
    why: '用户要的是"这是几"，不是一个概率分布。',
    how: '选最大的那个索引。对应的 exp(out[c]) 就是模型的"信心"。',
    code: 'predicted = argmax(out)',
    example: `out = [−6.27, −5.63, −3.10, −4.78, −7.58,
       −4.05, −7.10, −0.127, −6.01, −3.21]
         0      1     2     3     4
         5     6     7      8     9

最大值在索引 7 → predicted = 7
信心 = exp(−0.127) ≈ 0.881 = 88.1%

意思是：模型 88.1% 确定这是一个 7，剩下的 11.9% 分散在其它 9 个数字上（主要是 2 和 9，因为它们的 out 值相对较高）。`,
  },
]

const weights = [
  {
    name: 'W1',
    shape: '[784, 256]',
    role: '第一层模式探测器：每一列是一个神经元在 784 个像素上的权重',
    min: '−0.167',
    max: '+0.160',
    count: '200,704',
  },
  {
    name: 'b1',
    shape: '[256]',
    role: '256 个隐藏神经元各自的激活门槛',
    min: '−0.008',
    max: '+0.007',
    count: '256',
  },
  {
    name: 'W2',
    shape: '[256, 10]',
    role: '特征 → 数字的映射：每一列告诉模型"数字 c 喜欢哪些隐藏特征"',
    min: '−0.363',
    max: '+0.399',
    count: '2,560',
  },
  {
    name: 'b2',
    shape: '[10]',
    role: '10 个数字类别各自的偏好（prior）',
    min: '−0.008',
    max: '+0.010',
    count: '10',
  },
]

const repoUrl = 'https://github.com/mammgerL/c-tensor'

function navigate(path) {
  router.push(path)
}
</script>

<template>
  <div class="home-view">
    <section class="hero-section">
      <div class="hero-content">
        <h1 class="hero-title">
          <span class="title-text">C-Tensor</span>
        </h1>
        <p class="hero-subtitle">纯 C 实现的神经网络，从零理解 MNIST 手写数字识别</p>
        <p class="hero-desc">
          用纯 C 实现一个带自动微分的张量库，训练 784 → 256 → 10 的多层感知机识别手写数字。<br>
          这一页把神经网络的每一步拆开来讲 —— 先理解直觉，再看具体发生了什么。
        </p>
        <div class="hero-actions">
          <button class="btn-primary btn-lg" @click="navigate('/playground')">
            📐 开始计算演示
          </button>
          <button class="btn-secondary btn-lg" @click="navigate('/learn')">
            ⚙️ 查看 C 代码原理
          </button>
          <a
            class="btn-secondary btn-lg hero-link"
            :href="repoUrl"
            target="_blank"
            rel="noreferrer"
          >
            ↗ GitHub 仓库
          </a>
        </div>
      </div>
    </section>

    <section class="diagram-section">
      <h2 class="section-title">网络全景：784 → 256 → 10</h2>
      <p class="section-subtitle">
        下面每个圆圈是一个"神经元"，亮度代表激活程度。连线是权重。这就是模型看到一张 "7" 时的状态。
      </p>
      <div class="diagram-wrapper">
        <svg class="network-svg" viewBox="0 0 600 300" preserveAspectRatio="xMidYMid meet">
          <line v-for="(e, i) in diagramEdges" :key="'e-' + i"
            :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
            stroke="currentColor" :stroke-opacity="e.opacity" stroke-width="1" />
          <g v-for="(layer, li) in diagramNodes" :key="'l-' + li">
            <g v-for="(n, ni) in layer.nodes" :key="'n-' + ni">
              <circle :cx="layer.x" :cy="n.y" r="12"
                :fill="layer.color" :fill-opacity="0.12 + n.a * 0.88"
                :stroke="layer.color" :stroke-opacity="0.3" stroke-width="1.5" />
              <text v-if="li === 2" :x="layer.x + 22" :y="n.y + 4"
                fill="var(--color-text-light)" font-size="11" font-weight="600">
                {{ ni }}{{ ni === 7 ? ' ←' : '' }}
              </text>
            </g>
            <text :x="layer.x" y="295" text-anchor="middle" fill="var(--color-text)" font-size="12" font-weight="700">
              {{ layer.label }}
            </text>
            <text :x="layer.x" y="282" text-anchor="middle" fill="var(--color-text-light)" font-size="10">
              {{ layer.sub }}
            </text>
          </g>
          <text x="190" y="155" text-anchor="middle" fill="var(--color-text-light)" font-size="11" font-family="'SF Mono', monospace">
            W1 [784, 256]
          </text>
          <text x="410" y="155" text-anchor="middle" fill="var(--color-text-light)" font-size="11" font-family="'SF Mono', monospace">
            W2 [256, 10]
          </text>
        </svg>
      </div>
    </section>

    <section class="intuition-section">
      <h2 class="section-title">神经网络到底在做什么？</h2>
      <p class="section-subtitle">
        先抛开公式，理解直觉。下面 4 条借鉴了 3Blue1Brown《Neural Networks》系列的讲法。
      </p>
      <div class="intuition-grid">
        <div v-for="item in intuitions" :key="item.num" class="intuition-card">
          <div class="intuition-num">{{ item.num }}</div>
          <h3 class="intuition-title">{{ item.title }}</h3>
          <p class="intuition-body">{{ item.body }}</p>
          <!-- Card 01: neuron brightness row -->
          <div v-if="item.visual === 'neurons'" class="intuition-visual">
            <div class="neurons-demo">
              <div class="neurons-label">神经元亮度 = 激活值</div>
              <div class="neurons-row">
                <div v-for="(op, i) in [0.0, 0.15, 0.55, 0.92, 0.8, 0.35, 0.1, 0.0]" :key="i" class="neuron-item">
                  <span class="neuron-dot" :style="{ opacity: 0.1 + op * 0.9, background: '#6C63FF' }"></span>
                  <span class="neuron-val">{{ op.toFixed(1) }}</span>
                </div>
              </div>
            </div>
          </div>
          <!-- Card 02: weight pattern grid -->
          <div v-else-if="item.visual === 'weights'" class="intuition-visual">
            <div class="weight-pattern-demo">
              <div class="weight-pattern-grid">
                <div v-for="(row, r) in weightGrid" :key="r" class="wp-row">
                  <div v-for="(val, c) in row" :key="c" class="wp-cell"
                    :style="{ background: val > 0 ? `rgba(108,99,255,${val})` : val < 0 ? `rgba(255,82,82,${-val})` : 'rgba(0,0,0,0.04)' }"
                  />
                </div>
              </div>
              <div class="weight-pattern-legend">
                <span><span class="wp-swatch wp-pos"></span> 正权重：越亮越好</span>
                <span><span class="wp-swatch wp-neg"></span> 负权重：越暗越好</span>
              </div>
            </div>
          </div>
          <!-- Card 03: formula box -->
          <div v-else-if="item.visual === 'formula'" class="intuition-visual formula-box">{{ item.formulaText }}</div>
          <!-- Card 04: param number -->
          <div v-else-if="item.visual === 'params'" class="intuition-visual params-box">
            <span class="param-number">203,530</span>
            <span class="param-label">个可学习参数</span>
            <div class="param-breakdown">
              <div class="param-bar" title="W1: 200,704" style="flex: 200704; background: #6C63FF;"></div>
              <div class="param-bar" title="b1: 256" style="flex: 256; background: #FF6B9D;"></div>
              <div class="param-bar" title="W2: 2,560" style="flex: 2560; background: #00D2FF;"></div>
              <div class="param-bar" title="b2: 10" style="flex: 10; background: #FFB74D;"></div>
            </div>
            <div class="param-bar-legend">
              <span style="color: #6C63FF">W1 200,704</span>
              <span style="color: #FF6B9D">b1 256</span>
              <span style="color: #00D2FF">W2 2,560</span>
              <span style="color: #FFB74D">b2 10</span>
            </div>
          </div>
        </div>
      </div>
      <div class="intuition-honest">
        <span class="honest-badge">诚实地说</span>
        <p>
          理想情况下，第一层识别笔画边缘，第二层识别圈和弧线，最后组合成数字 —— 但实际训练出来的权重图更像噪声，
          神经元学到的是我们说不清楚的特征。而它就是有效。
          这就是深度学习让人又爱又困惑的地方。
        </p>
      </div>
    </section>

    <section class="matmul-section">
      <h2 class="section-title">深入理解：矩阵乘法到底在算什么？</h2>
      <p class="section-subtitle">
        用一个 [1,4] × [4,3] = [1,3] 的小例子拆解过程。真实网络是 [1,784] × [784,256] = [1,256]，原理完全一样，只是更大。
      </p>
      <div class="matmul-card">
        <!-- 三个矩阵并排 -->
        <div class="matmul-layout">
          <!-- x 向量 -->
          <div class="mat-block">
            <div class="mat-label">输入 x <span class="mat-shape">[1, 4]</span></div>
            <div class="mat-grid mat-row">
              <div v-for="(v, i) in matmulX" :key="'x-' + i"
                class="mat-cell mat-x"
                :class="{ 'mat-active': true }"
              >{{ v }}</div>
            </div>
          </div>

          <span class="mat-op">×</span>

          <!-- W 矩阵 -->
          <div class="mat-block">
            <div class="mat-label">权重 W <span class="mat-shape">[4, 3]</span></div>
            <div class="mat-grid mat-cols">
              <div v-for="(row, r) in matmulW" :key="'w-' + r" class="mat-grid-row">
                <div v-for="(v, c) in row" :key="'wc-' + c"
                  class="mat-cell mat-w"
                  :class="{ 'mat-col-hi': c === matmulHighlight }"
                >{{ v }}</div>
              </div>
            </div>
          </div>

          <span class="mat-op">=</span>

          <!-- h 向量 -->
          <div class="mat-block">
            <div class="mat-label">输出 h <span class="mat-shape">[1, 3]</span></div>
            <div class="mat-grid mat-row">
              <div v-for="(h, j) in matmulH" :key="'h-' + j"
                class="mat-cell mat-h"
                :class="{ 'mat-col-hi': j === matmulHighlight }"
                @mouseenter="matmulHighlight = j"
              >{{ h.sum.toFixed(2) }}</div>
            </div>
          </div>
        </div>

        <!-- 展开计算过程 -->
        <div class="matmul-detail">
          <div class="matmul-detail-tabs">
            <button v-for="j in 3" :key="j"
              :class="['matmul-tab', { active: matmulHighlight === j - 1 }]"
              @click="matmulHighlight = j - 1"
            >h[{{ j - 1 }}]</button>
          </div>
          <div class="matmul-computation">
            <div class="comp-title">
              h[{{ matmulHighlight }}] = x 的每一项 × W 第 {{ matmulHighlight }} 列的对应项，然后求和：
            </div>
            <div class="comp-terms">
              <span v-for="(t, i) in matmulH[matmulHighlight].terms" :key="i" class="comp-term">
                <span class="comp-operand">{{ matmulX[i] }}</span>
                <span class="comp-times">×</span>
                <span class="comp-operand">{{ matmulW[i][matmulHighlight] }}</span>
                <span class="comp-eq">=</span>
                <span class="comp-result" :class="{ pos: t >= 0, neg: t < 0 }">{{ t >= 0 ? '+' : '' }}{{ t.toFixed(2) }}</span>
              </span>
            </div>
            <div class="comp-sum">
              求和：{{ matmulH[matmulHighlight].terms.map(t => (t >= 0 ? '+' : '') + t.toFixed(2)).join(' ') }}
              = <strong>{{ matmulH[matmulHighlight].sum.toFixed(2) }}</strong>
            </div>
          </div>
        </div>

        <!-- 对应到真实网络 -->
        <div class="matmul-real">
          <div class="matmul-real-arrow">↓ 放大到真实网络</div>
          <div class="matmul-real-grid">
            <div class="real-item">
              <span class="real-label">输入</span>
              <span class="real-val">[1, 4] → <strong>[1, 784]</strong></span>
              <span class="real-desc">4 个像素 → 28×28 = 784 个像素</span>
            </div>
            <div class="real-item">
              <span class="real-label">权重</span>
              <span class="real-val">[4, 3] → <strong>[784, 256]</strong></span>
              <span class="real-desc">3 个神经元 → 256 个神经元，每个做 784 次乘加</span>
            </div>
            <div class="real-item">
              <span class="real-label">输出</span>
              <span class="real-val">[1, 3] → <strong>[1, 256]</strong></span>
              <span class="real-desc">3 个特征 → 256 个特征</span>
            </div>
          </div>
        </div>

        <!-- 偏置 + ReLU 计算举例 -->
        <div class="bias-relu-box">
          <h4 class="bias-title">第二步：加偏置 + ReLU 激活</h4>
          <p class="bias-desc">矩阵乘法只完成了"加权求和"，还需要加偏置再过 ReLU，才得到隐藏层的最终输出。</p>

          <div class="bias-steps">
            <!-- matmul result -->
            <div class="bias-step">
              <span class="bias-label">矩阵乘法结果 h</span>
              <div class="bias-vec">
                <span v-for="(h, j) in matmulH" :key="'bh-' + j" class="bias-cell">{{ h.sum.toFixed(2) }}</span>
              </div>
            </div>

            <div class="bias-op">+</div>

            <!-- bias -->
            <div class="bias-step">
              <span class="bias-label">偏置 b</span>
              <div class="bias-vec">
                <span v-for="(b, j) in biasB" :key="'bb-' + j" class="bias-cell bias-b">{{ b >= 0 ? '+' : '' }}{{ b.toFixed(2) }}</span>
              </div>
            </div>

            <div class="bias-op">=</div>

            <!-- after bias -->
            <div class="bias-step">
              <span class="bias-label">加偏置后</span>
              <div class="bias-vec">
                <span v-for="(v, j) in afterBias" :key="'ab-' + j" class="bias-cell" :class="{ 'neg': v < 0 }">{{ v.toFixed(2) }}</span>
              </div>
            </div>

            <div class="bias-op relu-arrow">→ ReLU →</div>

            <!-- after ReLU -->
            <div class="bias-step">
              <span class="bias-label">最终输出</span>
              <div class="bias-vec">
                <span v-for="(v, j) in afterRelu" :key="'ar-' + j" class="bias-cell" :class="{ 'zero': v === 0, 'pos': v > 0 }">{{ v.toFixed(2) }}</span>
              </div>
            </div>
          </div>

          <div class="bias-explain">
            <p>ReLU 把负数变成 0 — 这就是"激活"：只有正向信号才能传递到下一层。</p>
            <p>3 个输出中有 2 个被抑制，只有 <strong>h[2] = 0.44</strong> 存活下来。</p>
          </div>
        </div>
      </div>
    </section>

    <section class="weights-section">
      <h2 class="section-title">打开黑盒：网络里到底存了什么？</h2>
      <p class="section-subtitle">
        "训练好的网络"其实就是下面这 4 个张量里的 203,530 个具体浮点数。下面是本项目真实训练后的数据。
      </p>
      <div class="weights-grid">
        <div v-for="w in weights" :key="w.name" class="weight-card">
          <div class="weight-head">
            <span class="weight-name">{{ w.name }}</span>
            <span class="weight-shape">{{ w.shape }}</span>
          </div>
          <p class="weight-role">{{ w.role }}</p>
          <div class="weight-stats">
            <div class="weight-stat">
              <span class="stat-label">数值范围</span>
              <span class="stat-value">{{ w.min }} ~ {{ w.max }}</span>
            </div>
            <div class="weight-stat">
              <span class="stat-label">参数个数</span>
              <span class="stat-value">{{ w.count }}</span>
            </div>
          </div>
        </div>
      </div>
      <div class="weights-notes">
        <h3 class="notes-title">从这组数字里能看出什么？</h3>
        <ul>
          <li><strong>b1、b2 都非常接近 0</strong>（最大也就 ±0.01）—— 训练出来的神经元"挑剔程度"其实都差不多，偏置只做微调。</li>
          <li><strong>W1 的值域 (±0.16) 比 W2 (±0.40) 小</strong> —— Kaiming 初始化按 <code>√(2/fan_in)</code> 缩放，784 个输入的扇入让 W1 一开始就被压得比较小，训练过程也保持了这个量级。</li>
          <li><strong>W1 每一列排回 28×28 后，并不像边缘或笔画检测器</strong> —— 更像是有一定结构的噪声。这就是 3Blue1Brown 系列里说的"愿望 vs 现实"。</li>
          <li>想亲眼看看这些数字怎么工作？<a href="#" @click.prevent="navigate('/playground')">去 Playground</a> 画一个数字，每一个神经元被激活了多少都是实时计算出来的。</li>
        </ul>
      </div>

      <div class="file-format">
        <h3 class="notes-title">mnist_mlp.bin 文件是怎么组织的？</h3>
        <p class="format-desc">
          训练完成后，模型以二进制格式保存到 <code>mnist_mlp.bin</code> 文件中。整个文件由一个固定大小的文件头 + 四块连续的 float 数据组成，没有任何框架依赖，用 C 的 <code>fread</code> 就能直接读取。
        </p>
        <div class="format-structure">
          <div class="format-block header">
            <span class="block-label">文件头</span>
            <span class="block-size">28 bytes</span>
            <div class="block-fields">
              <span class="field"><code>magic</code> <span class="field-val">0x314C504D ("MLP1")</span></span>
              <span class="field"><code>version</code> <span class="field-val">1</span></span>
              <span class="field"><code>w1_rows, w1_cols</code> <span class="field-val">784, 256</span></span>
              <span class="field"><code>b1_len</code> <span class="field-val">256</span></span>
              <span class="field"><code>w2_rows, w2_cols</code> <span class="field-val">256, 10</span></span>
              <span class="field"><code>b2_len</code> <span class="field-val">10</span></span>
            </div>
          </div>
          <div class="format-arrow">↓</div>
          <div class="format-block data">
            <span class="block-label">W1 数据</span>
            <span class="block-size">802,816 bytes</span>
            <span class="block-detail">200,704 × float32，按行主序排列</span>
          </div>
          <div class="format-arrow">↓</div>
          <div class="format-block data">
            <span class="block-label">b1 数据</span>
            <span class="block-size">1,024 bytes</span>
            <span class="block-detail">256 × float32</span>
          </div>
          <div class="format-arrow">↓</div>
          <div class="format-block data">
            <span class="block-label">W2 数据</span>
            <span class="block-size">10,240 bytes</span>
            <span class="block-detail">2,560 × float32</span>
          </div>
          <div class="format-arrow">↓</div>
          <div class="format-block data">
            <span class="block-label">b2 数据</span>
            <span class="block-size">40 bytes</span>
            <span class="block-detail">10 × float32</span>
          </div>
        </div>
        <p class="format-summary">
          总大小约 <strong>814,148 bytes（~795 KB）</strong>。文件头记录了每个张量的形状，加载时先读头再按尺寸分配内存、读数据。这种设计让模型文件可以独立于训练代码被加载和验证。
        </p>
      </div>
    </section>

    <section class="principles-section">
      <h2 class="section-title">跟着一张 "7" 走一遍网络</h2>
      <p class="section-subtitle">
        从像素到答案，前向传播的每一步都在解决一个具体问题。左边是"为什么"，右边是"怎么做"。
      </p>
      <div class="principles-list">
        <article v-for="item in principles" :key="item.num" class="principle-card">
          <header class="principle-head">
            <span class="principle-num">{{ item.num }}</span>
            <h3 class="principle-title">{{ item.title }}</h3>
            <span class="principle-shape">{{ item.shape }}</span>
          </header>
          <div class="principle-body">
            <div class="principle-col">
              <span class="col-label">问题</span>
              <p>{{ item.why }}</p>
            </div>
            <div class="principle-col">
              <span class="col-label">做法</span>
              <p>{{ item.how }}</p>
            </div>
          </div>
          <code class="principle-code">{{ item.code }}</code>

          <!-- ReLU function graph for step 04 -->
          <div v-if="item.num === '04'" class="principle-figure">
            <svg class="relu-svg" viewBox="0 0 200 100" preserveAspectRatio="xMidYMid meet">
              <line x1="10" y1="50" x2="190" y2="50" stroke="var(--color-border, #ddd)" stroke-width="1"/>
              <line x1="100" y1="8" x2="100" y2="92" stroke="var(--color-border, #ddd)" stroke-width="1"/>
              <polyline points="10,50 100,50 190,8" stroke="#6C63FF" stroke-width="2.5" fill="none" stroke-linecap="round"/>
              <circle cx="100" cy="50" r="3.5" fill="#6C63FF"/>
              <text x="22" y="44" fill="var(--color-text-light)" font-size="9">y = 0</text>
              <text x="160" y="22" fill="#6C63FF" font-size="10" font-weight="700">y = x</text>
              <text x="54" y="64" fill="var(--color-text-light)" font-size="9">x &lt; 0</text>
              <text x="130" y="64" fill="var(--color-text-light)" font-size="9">x &gt; 0</text>
            </svg>
            <div class="figure-caption">ReLU(x) = max(0, x) — 左边全部归零，右边原样保留</div>
          </div>

          <!-- Softmax probability bars for step 06 -->
          <div v-if="item.num === '07'" class="principle-figure">
            <div class="prob-bars">
              <div v-for="bar in softmaxBars" :key="bar.digit" class="prob-bar-row">
                <span class="prob-digit" :class="{ highlight: bar.digit === 7 }">{{ bar.digit }}</span>
                <div class="prob-bar-track">
                  <div class="prob-bar-fill"
                    :style="{ width: (bar.prob * 100) + '%', background: bar.digit === 7 ? '#6C63FF' : 'var(--color-border, #ddd)' }"
                  />
                </div>
                <span class="prob-pct" :class="{ highlight: bar.digit === 7 }">{{ (bar.prob * 100).toFixed(1) }}%</span>
              </div>
            </div>
            <div class="figure-caption">模型对 "7" 的输出概率分布 — 88.1% 的信心集中在正确类别</div>
          </div>

          <details v-if="item.example" class="principle-example">
            <summary>举个具体例子 ▾</summary>
            <pre>{{ item.example }}</pre>
          </details>
        </article>
      </div>
    </section>

    <section class="features-section">
      <h2 class="section-title">选择你要探索的方向</h2>
      <div class="features-grid">
        <div
          v-for="feature in features"
          :key="feature.path"
          class="feature-card"
          @click="navigate(feature.path)"
        >
          <div class="feature-icon">{{ feature.icon }}</div>
          <h3 class="feature-title">{{ feature.title }}</h3>
          <p class="feature-desc">{{ feature.desc }}</p>
          <div class="feature-arrow" :style="{ color: feature.color }">→</div>
        </div>
      </div>
    </section>

  </div>
</template>

<style scoped>
.home-view {
  max-width: 1100px;
  margin: 0 auto;
  padding: 40px 24px;
}

.hero-section {
  margin-bottom: 60px;
}

.hero-title {
  margin-bottom: 12px;
}

.title-text {
  font-size: 48px;
  font-weight: 800;
  background: linear-gradient(135deg, #6C63FF, #00D2FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 20px;
  font-weight: 600;
  color: var(--color-text-light);
  margin-bottom: 16px;
}

.hero-desc {
  font-size: 15px;
  line-height: 1.7;
  color: var(--color-text-light);
  margin-bottom: 28px;
}

.hero-actions {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.btn-lg {
  padding: 14px 28px;
  font-size: 16px;
}

.hero-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  text-decoration: none;
}

.section-title {
  font-size: 24px;
  font-weight: 800;
  text-align: center;
  margin-bottom: 28px;
}

.section-subtitle {
  text-align: center;
  font-size: 14px;
  color: var(--color-text-light);
  margin: -20px auto 28px;
  max-width: 560px;
  line-height: 1.6;
}

/* ── Network diagram ── */
.diagram-section {
  margin-bottom: 60px;
}

.diagram-wrapper {
  background: var(--color-card);
  border-radius: 16px;
  padding: 20px 10px 10px;
  box-shadow: var(--shadow-sm);
  overflow: hidden;
}

.network-svg {
  width: 100%;
  max-width: 700px;
  height: auto;
  margin: 0 auto;
  display: block;
  color: var(--color-text-light);
}

/* ── Intuition visuals ── */
.neurons-demo {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.neurons-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: var(--color-text-light);
  text-transform: uppercase;
}

.neurons-row {
  display: flex;
  gap: 8px;
  align-items: center;
  padding-top: 6px;
}

.neuron-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.neuron-dot {
  width: 22px;
  height: 22px;
  border-radius: 50%;
}

.neuron-val {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 10px;
  color: var(--color-text-light);
}

/* Weight pattern grid */
.weight-pattern-demo {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.weight-pattern-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 4px;
  background: var(--color-bg);
  border-radius: 6px;
}

.wp-row {
  display: flex;
  gap: 2px;
}

.wp-cell {
  width: 20px;
  height: 20px;
  border-radius: 3px;
}

.weight-pattern-legend {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 11.5px;
  color: var(--color-text-light);
}

.wp-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  vertical-align: middle;
  margin-right: 5px;
}

.wp-pos { background: rgba(108, 99, 255, 0.7); }
.wp-neg { background: rgba(255, 82, 82, 0.7); }

/* Param breakdown bar */
.param-breakdown {
  display: flex;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  margin-top: 10px;
  gap: 2px;
}

.param-bar {
  border-radius: 2px;
  min-width: 2px;
}

.param-bar-legend {
  display: flex;
  gap: 10px;
  margin-top: 6px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 10px;
  font-weight: 600;
  flex-wrap: wrap;
}

/* ── Principle figures (ReLU, Softmax) ── */
.principle-figure {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px dashed var(--color-border, rgba(0,0,0,0.08));
}

.relu-svg {
  width: 100%;
  max-width: 260px;
  height: auto;
  display: block;
  margin: 0 auto;
}

.figure-caption {
  text-align: center;
  font-size: 11.5px;
  color: var(--color-text-light);
  margin-top: 8px;
  line-height: 1.5;
}

/* Probability bars */
.prob-bars {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.prob-bar-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.prob-digit {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  font-weight: 600;
  width: 14px;
  text-align: right;
  color: var(--color-text-light);
}

.prob-digit.highlight {
  color: #6C63FF;
  font-weight: 800;
}

.prob-bar-track {
  flex: 1;
  height: 10px;
  background: var(--color-bg);
  border-radius: 5px;
  overflow: hidden;
}

.prob-bar-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.3s ease;
}

.prob-pct {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px;
  color: var(--color-text-light);
  width: 42px;
  text-align: right;
}

.prob-pct.highlight {
  color: #6C63FF;
  font-weight: 700;
}

/* ── Matmul deep-dive ── */
.matmul-section {
  margin-bottom: 60px;
}

.matmul-card {
  background: var(--color-card);
  border-radius: 16px;
  padding: 28px 24px;
  box-shadow: var(--shadow-sm);
}

.matmul-layout {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 24px;
}

.mat-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.mat-label {
  font-size: 13px;
  font-weight: 700;
  color: var(--color-text);
}

.mat-shape {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-light);
  margin-left: 4px;
}

.mat-op {
  font-size: 22px;
  font-weight: 700;
  color: var(--color-text-light);
  padding: 0 4px;
}

.mat-grid-row {
  display: flex;
  gap: 3px;
}

.mat-cols {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.mat-row {
  display: flex;
  gap: 3px;
}

.mat-cell {
  width: 48px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 5px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  font-weight: 600;
  transition: all 0.15s ease;
}

.mat-x {
  background: rgba(108, 99, 255, 0.12);
  color: #6C63FF;
}

.mat-w {
  background: rgba(0, 0, 0, 0.04);
  color: var(--color-text);
}

.mat-w.mat-col-hi {
  background: rgba(255, 107, 157, 0.15);
  color: #FF6B9D;
}

.mat-h {
  background: rgba(0, 210, 255, 0.12);
  color: #00A5CC;
  cursor: pointer;
}

.mat-h.mat-col-hi {
  background: rgba(0, 210, 255, 0.25);
  box-shadow: 0 0 0 2px rgba(0, 210, 255, 0.4);
}

/* Computation detail */
.matmul-detail {
  background: var(--color-bg);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 20px;
}

.matmul-detail-tabs {
  display: flex;
  gap: 6px;
  margin-bottom: 14px;
}

.matmul-tab {
  padding: 5px 14px;
  border-radius: 6px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  font-weight: 600;
  background: var(--color-card);
  color: var(--color-text-light);
  cursor: pointer;
  transition: all 0.15s;
}

.matmul-tab.active {
  background: rgba(0, 210, 255, 0.15);
  color: #00A5CC;
  box-shadow: 0 0 0 1.5px rgba(0, 210, 255, 0.3);
}

.comp-title {
  font-size: 13px;
  color: var(--color-text-light);
  margin-bottom: 12px;
  line-height: 1.5;
}

.comp-terms {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 14px;
}

.comp-term {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
}

.comp-operand {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 42px;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 600;
}

.comp-term:nth-child(odd) .comp-operand:first-child {
  background: rgba(108, 99, 255, 0.1);
  color: #6C63FF;
}

.comp-operand:nth-child(3) {
  background: rgba(255, 107, 157, 0.1);
  color: #FF6B9D;
}

.comp-times, .comp-eq {
  color: var(--color-text-light);
  font-weight: 400;
}

.comp-result {
  font-weight: 700;
  min-width: 48px;
  text-align: right;
}

.comp-result.pos { color: #4CAF50; }
.comp-result.neg { color: #FF5252; }

.comp-sum {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--color-text);
  padding-top: 10px;
  border-top: 1px dashed var(--color-border, rgba(0,0,0,0.08));
  line-height: 1.6;
}

.comp-sum strong {
  color: #00A5CC;
  font-size: 15px;
}

/* Real network scale comparison */
.matmul-real {
  text-align: center;
}

.matmul-real-arrow {
  font-size: 13px;
  font-weight: 700;
  color: var(--color-primary);
  margin-bottom: 12px;
}

.matmul-real-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.real-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px;
  background: var(--color-bg);
  border-radius: 10px;
}

.real-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: var(--color-text-light);
  text-transform: uppercase;
}

.real-val {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--color-text);
}

.real-val strong {
  color: var(--color-primary);
}

.real-desc {
  font-size: 11.5px;
  color: var(--color-text-light);
  line-height: 1.5;
}

@media (max-width: 768px) {
  .matmul-layout {
    flex-direction: column;
    gap: 10px;
  }

  .matmul-real-grid {
    grid-template-columns: 1fr;
  }

  .mat-op {
    font-size: 18px;
  }
}

.bias-relu-box {
  margin-top: 32px;
  padding: 24px 28px;
  background: var(--color-card);
  border-radius: 14px;
  border: 1px solid var(--color-border, #e5e7eb);
}

.bias-title {
  font-size: 17px;
  font-weight: 700;
  margin: 0 0 6px;
}

.bias-desc {
  font-size: 14px;
  color: var(--color-text-light);
  margin: 0 0 20px;
}

.bias-steps {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.bias-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.bias-label {
  font-size: 12px;
  color: var(--color-text-light);
  font-weight: 600;
}

.bias-vec {
  display: flex;
  gap: 4px;
}

.bias-cell {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 52px;
  padding: 6px 8px;
  border-radius: 8px;
  font-family: 'Fira Code', 'Cascadia Code', monospace;
  font-size: 14px;
  font-weight: 600;
  background: #f0f0ff;
  color: #333;
}

.bias-cell.bias-b {
  background: #fff3e0;
  color: #e65100;
}

.bias-cell.neg {
  background: #ffebee;
  color: #c62828;
}

.bias-cell.zero {
  background: #f5f5f5;
  color: #999;
}

.bias-cell.pos {
  background: #e8f5e9;
  color: #2e7d32;
  font-weight: 700;
}

.bias-op {
  font-size: 20px;
  font-weight: 700;
  color: var(--color-text-light);
  padding: 0 2px;
}

.bias-op.relu-arrow {
  font-size: 14px;
  color: #6C63FF;
  white-space: nowrap;
}

.bias-explain {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--color-border, #e5e7eb);
  font-size: 14px;
  color: var(--color-text-light);
  line-height: 1.7;
}

.bias-explain p {
  margin: 0 0 4px;
}

@media (max-width: 700px) {
  .bias-steps {
    flex-direction: column;
    gap: 8px;
  }

  .bias-op {
    font-size: 16px;
  }

  .bias-cell {
    min-width: 44px;
    font-size: 12px;
  }
}

.intuition-section {
  margin-bottom: 60px;
}

.intuition-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.intuition-card {
  position: relative;
  background: var(--color-card);
  border-radius: 16px;
  padding: 28px 26px 24px;
  box-shadow: var(--shadow-sm);
  transition: var(--transition);
  display: flex;
  flex-direction: column;
}

.intuition-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.intuition-num {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  font-weight: 700;
  color: var(--color-primary);
  letter-spacing: 1px;
  margin-bottom: 10px;
}

.intuition-title {
  font-size: 17px;
  font-weight: 700;
  margin-bottom: 12px;
  line-height: 1.4;
}

.intuition-body {
  font-size: 14px;
  line-height: 1.75;
  color: var(--color-text-light);
  margin-bottom: 16px;
  flex: 1;
}

.intuition-visual {
  margin-top: auto;
  padding-top: 12px;
  border-top: 1px dashed var(--color-border, rgba(0,0,0,0.08));
}

.neurons-row {
  display: flex;
  gap: 6px;
  align-items: center;
  justify-content: flex-start;
  padding-top: 14px;
}

.neuron-dot {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--color-primary);
}

.formula-box {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--color-text);
  background: var(--color-bg);
  padding: 10px 12px;
  border-radius: 8px;
  margin-top: 12px;
  text-align: center;
}

.params-box {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding-top: 14px;
}

.param-number {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 26px;
  font-weight: 800;
  background: linear-gradient(135deg, #6C63FF, #00D2FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
}

.param-label {
  font-size: 12px;
  color: var(--color-text-light);
}

.intuition-honest {
  display: flex;
  gap: 14px;
  align-items: flex-start;
  background: rgba(255, 107, 157, 0.06);
  border-left: 3px solid #FF6B9D;
  border-radius: 10px;
  padding: 16px 18px;
}

.intuition-honest p {
  font-size: 14px;
  line-height: 1.75;
  color: var(--color-text-light);
  margin: 0;
}

.honest-badge {
  flex-shrink: 0;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: #FF6B9D;
  background: rgba(255, 107, 157, 0.12);
  padding: 4px 10px;
  border-radius: 6px;
  margin-top: 2px;
}

.principles-section {
  margin-bottom: 60px;
}

.principles-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.principle-card {
  background: var(--color-card);
  border-radius: 14px;
  padding: 22px 24px 18px;
  box-shadow: var(--shadow-sm);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
}

.principle-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, #6C63FF, #00D2FF);
  opacity: 0;
  transition: var(--transition);
}

.principle-card:hover {
  transform: translateX(3px);
  box-shadow: var(--shadow-lg);
}

.principle-card:hover::before {
  opacity: 1;
}

.principle-head {
  display: flex;
  align-items: baseline;
  gap: 14px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}

.principle-num {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  font-weight: 700;
  color: var(--color-primary);
  letter-spacing: 1px;
  flex-shrink: 0;
}

.principle-title {
  font-size: 16px;
  font-weight: 700;
  margin: 0;
  flex: 1;
  line-height: 1.4;
}

.principle-shape {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 3px 8px;
  border-radius: 5px;
  flex-shrink: 0;
}

.principle-body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-bottom: 14px;
}

.principle-col {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.principle-col p {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-light);
  margin: 0;
}

.col-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--color-primary);
  text-transform: uppercase;
}

.principle-col:nth-child(2) .col-label {
  color: #FF6B9D;
}

.principle-code {
  display: block;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12.5px;
  color: var(--color-text);
  background: var(--color-bg);
  padding: 8px 12px;
  border-radius: 6px;
  overflow-x: auto;
  white-space: nowrap;
}

.principle-example {
  margin-top: 10px;
  border-radius: 8px;
  background: rgba(108, 99, 255, 0.04);
  border: 1px solid rgba(108, 99, 255, 0.15);
  overflow: hidden;
}

.principle-example summary {
  padding: 8px 14px;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--color-primary);
  cursor: pointer;
  user-select: none;
  list-style: none;
  transition: background 0.15s;
}

.principle-example summary::-webkit-details-marker {
  display: none;
}

.principle-example summary:hover {
  background: rgba(108, 99, 255, 0.08);
}

.principle-example[open] summary {
  border-bottom: 1px solid rgba(108, 99, 255, 0.15);
  background: rgba(108, 99, 255, 0.08);
}

.weights-section {
  margin-bottom: 60px;
}

.weights-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
  margin-bottom: 24px;
}

.weight-card {
  background: var(--color-card);
  border-radius: 12px;
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
  transition: var(--transition);
}

.weight-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.weight-head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 8px;
}

.weight-name {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 18px;
  font-weight: 800;
  color: var(--color-primary);
}

.weight-shape {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 2px 8px;
  border-radius: 5px;
}

.weight-role {
  font-size: 13px;
  line-height: 1.65;
  color: var(--color-text-light);
  margin: 0 0 12px 0;
}

.weight-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  padding-top: 12px;
  border-top: 1px dashed rgba(0, 0, 0, 0.08);
}

.weight-stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: var(--color-text-light);
  text-transform: uppercase;
}

.stat-value {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  font-weight: 700;
  color: var(--color-text);
}

.weights-notes {
  background: rgba(0, 210, 255, 0.05);
  border-left: 3px solid #00D2FF;
  border-radius: 10px;
  padding: 18px 22px;
}

.notes-title {
  font-size: 14px;
  font-weight: 700;
  margin: 0 0 12px 0;
  color: var(--color-text);
}

.weights-notes ul {
  margin: 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.weights-notes li {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-light);
}

.weights-notes code {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  background: var(--color-bg);
  padding: 1px 6px;
  border-radius: 4px;
  color: var(--color-text);
}

.weights-notes a {
  color: var(--color-primary);
  text-decoration: none;
  font-weight: 600;
}

.weights-notes a:hover {
  text-decoration: underline;
}

.file-format {
  background: rgba(108, 99, 255, 0.04);
  border-left: 3px solid #6C63FF;
  border-radius: 10px;
  padding: 18px 22px;
  margin-top: 14px;
}

.format-desc {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-light);
  margin: 0 0 16px 0;
}

.format-desc code {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  background: var(--color-bg);
  padding: 1px 6px;
  border-radius: 4px;
  color: var(--color-text);
}

.format-structure {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  margin-bottom: 16px;
}

.format-block {
  width: 100%;
  max-width: 520px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 13px;
}

.format-block.header {
  background: var(--color-card);
  border: 1px solid rgba(108, 99, 255, 0.2);
  flex-wrap: wrap;
}

.format-block.data {
  background: var(--color-card);
  border: 1px solid rgba(0, 210, 255, 0.15);
}

.block-label {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-weight: 700;
  font-size: 14px;
  color: var(--color-primary);
  white-space: nowrap;
}

.format-block.data .block-label {
  color: #00D2FF;
}

.block-size {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 2px 8px;
  border-radius: 5px;
  white-space: nowrap;
}

.block-detail {
  font-size: 12px;
  color: var(--color-text-light);
}

.block-fields {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  width: 100%;
  margin-top: 6px;
  padding-top: 8px;
  border-top: 1px dashed rgba(108, 99, 255, 0.15);
}

.field {
  font-size: 12px;
  color: var(--color-text-light);
}

.field code {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px;
  color: var(--color-text);
}

.field-val {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px;
  color: var(--color-primary);
}

.format-arrow {
  font-size: 18px;
  color: var(--color-text-light);
  opacity: 0.4;
  line-height: 1;
}

.format-summary {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-light);
  margin: 0;
}

.format-summary strong {
  color: var(--color-text);
}

.principle-example pre {
  margin: 0;
  padding: 12px 14px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.7;
  color: var(--color-text);
  background: transparent;
  white-space: pre;
  overflow-x: auto;
}

.features-section {
  margin-bottom: 60px;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.feature-card {
  background: var(--color-card);
  border-radius: 16px;
  padding: 28px 24px;
  text-align: center;
  cursor: pointer;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
}

.feature-card:hover {
  transform: translateY(-6px);
  box-shadow: var(--shadow-lg);
}

.feature-icon {
  font-size: 40px;
  margin-bottom: 14px;
  display: block;
}

.feature-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 8px;
}

.feature-desc {
  font-size: 13px;
  color: var(--color-text-light);
  line-height: 1.6;
  margin-bottom: 14px;
}

.feature-arrow {
  font-size: 22px;
  font-weight: 700;
  transition: var(--transition);
}

.feature-card:hover .feature-arrow {
  transform: translateX(6px);
}

@media (max-width: 768px) {
  .features-grid {
    grid-template-columns: 1fr;
  }

  .intuition-grid {
    grid-template-columns: 1fr;
  }

  .weights-grid {
    grid-template-columns: 1fr;
  }

  .principle-body {
    grid-template-columns: 1fr;
    gap: 12px;
  }

  .weight-pattern-demo {
    justify-content: center;
  }

  .title-text {
    font-size: 36px;
  }
}
</style>
