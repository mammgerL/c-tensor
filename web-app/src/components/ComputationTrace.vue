<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  step: { type: Number, default: -1 },
  result: { type: Object, default: null },
})

const activeNeuron = ref(0)
const neuronWeights = ref(null)
const isLoadingWeights = ref(false)

const HIDDEN_SIZE = 256

// ─── Step 0: Input ───
const inputStats = computed(() => {
  if (!props.result?.pixels) return null
  const pixels = props.result.pixels
  const total = pixels.length
  const activePixels = pixels.filter(v => v > -0.5)
  const brightPixels = pixels.filter(v => v > 0)
  const maxVal = Math.max(...pixels)
  const mean = pixels.reduce((a, b) => a + b, 0) / total

  return { total, activeCount: activePixels.length, brightCount: brightPixels.length, maxVal, mean }
})

// Concrete normalization examples: pick 3 representative pixels
const normExamples = computed(() => {
  if (!props.result?.pixels) return []
  const pixels = props.result.pixels
  // Find brightest, a mid-tone, and a background pixel
  let brightIdx = 0, midIdx = -1, bgIdx = -1
  let brightVal = -2
  for (let i = 0; i < pixels.length; i++) {
    if (pixels[i] > brightVal) { brightVal = pixels[i]; brightIdx = i }
  }
  for (let i = 0; i < pixels.length; i++) {
    if (pixels[i] > -0.3 && pixels[i] < 0.3 && midIdx === -1) midIdx = i
    if (pixels[i] <= -0.95 && bgIdx === -1) bgIdx = i
  }
  if (midIdx === -1) midIdx = Math.floor(pixels.length / 2)
  if (bgIdx === -1) bgIdx = 0

  return [brightIdx, midIdx, bgIdx].map(i => {
    const v = pixels[i]
    const gray = Math.round(((v + 1) / 2) * 255)
    return {
      index: i,
      row: Math.floor(i / 28),
      col: i % 28,
      value: v,
      gray,
      label: v > 0.5 ? '笔画' : v > -0.5 ? '边缘' : '背景',
    }
  })
})

const topActivePixels = computed(() => {
  if (!props.result?.pixels) return []
  const pixels = props.result.pixels
  const entries = []
  for (let i = 0; i < pixels.length; i++) {
    if (pixels[i] > -0.5) {
      entries.push({
        index: i, row: Math.floor(i / 28), col: i % 28,
        value: pixels[i], normalized: (pixels[i] + 1) / 2,
      })
    }
  }
  entries.sort((a, b) => b.value - a.value)
  return entries.slice(0, 10)
})

const inputVectorRows = computed(() => {
  if (!props.result?.pixels) return []
  return props.result.pixels.map((v, i) => ({
    index: i,
    row: Math.floor(i / 28),
    col: i % 28,
    value: v,
    absValue: Math.abs(v),
  }))
})

// ─── Step 1-3: Hidden layer neurons ───
const topContributorNeurons = computed(() => {
  if (!props.result?.matmul1) return []
  const matmul1 = props.result.matmul1

  const neurons = []
  for (let h = 0; h < HIDDEN_SIZE; h++) {
    const preRelu = props.result.pre_relu?.[h] ?? 0
    const hidden = props.result.hidden?.[h] ?? 0
    neurons.push({
      index: h,
      matmul1Value: matmul1[h],
      absValue: Math.abs(matmul1[h]),
      bias1Value: preRelu - matmul1[h],
      preReluValue: preRelu,
      hiddenValue: hidden,
      reluActivated: hidden > 0,
    })
  }
  neurons.sort((a, b) => b.absValue - a.absValue)
  return neurons.slice(0, 8)
})

const selectedNeuron = computed(() => {
  return topContributorNeurons.value[activeNeuron.value] ?? null
})

async function fetchWeights(neuronIdx) {
  isLoadingWeights.value = true
  neuronWeights.value = null
  try {
    const res = await fetch(`/api/weights?layer=1&neuron=${neuronIdx}`)
    if (res.ok) neuronWeights.value = await res.json()
  } catch (e) {
    console.error('Failed to fetch weights:', e)
  } finally {
    isLoadingWeights.value = false
  }
}

watch([() => activeNeuron.value, () => props.step], ([neuronIdx, step]) => {
  if (step === 1 && topContributorNeurons.value.length > 0) {
    const realIdx = topContributorNeurons.value[neuronIdx]?.index ?? 0
    fetchWeights(realIdx)
  }
}, { immediate: true })

const pixelContributions = computed(() => {
  if (!props.result?.pixels || !neuronWeights.value?.weights) return []
  const pixels = props.result.pixels
  const weights = neuronWeights.value.weights
  const contribs = []
  for (let i = 0; i < 784; i++) {
    const contrib = pixels[i] * weights[i]
    contribs.push({ index: i, pixel: pixels[i], weight: weights[i], contrib, absContrib: Math.abs(contrib) })
  }
  contribs.sort((a, b) => b.absContrib - a.absContrib)
  return contribs.slice(0, 12)
})

// ─── Step 2: Bias stats ───
const biasStats = computed(() => {
  if (!topContributorNeurons.value.length) return null
  const biases = topContributorNeurons.value.map(n => n.bias1Value)
  const allBiases = []
  for (let h = 0; h < HIDDEN_SIZE; h++) {
    const preRelu = props.result?.pre_relu?.[h] ?? 0
    const matmul1 = props.result?.matmul1?.[h] ?? 0
    allBiases.push(preRelu - matmul1)
  }
  // Count how many neurons changed sign after bias
  let signFlips = 0
  for (let h = 0; h < HIDDEN_SIZE; h++) {
    const m = props.result?.matmul1?.[h] ?? 0
    const p = props.result?.pre_relu?.[h] ?? 0
    if ((m > 0 && p <= 0) || (m <= 0 && p > 0)) signFlips++
  }
  return {
    avgBias: allBiases.reduce((a, b) => a + b, 0) / allBiases.length,
    maxBias: Math.max(...allBiases),
    minBias: Math.min(...allBiases),
    signFlips,
  }
})

// ─── Step 3: ReLU stats ───
const reluFullStats = computed(() => {
  if (!props.result?.pre_relu || !props.result?.hidden) return null
  const pre = props.result.pre_relu
  const post = props.result.hidden
  const active = post.filter(v => v > 0).length
  const dead = HIDDEN_SIZE - active
  const maxPre = Math.max(...pre)
  const minPre = Math.min(...pre)
  const totalEnergy = post.reduce((a, b) => a + b, 0)
  // Distribution: count in bins
  const bins = Array(10).fill(0)
  const range = maxPre - minPre
  for (const v of pre) {
    const bin = Math.min(9, Math.floor(((v - minPre) / range) * 10))
    bins[bin]++
  }
  // Where does 0 fall in the bins?
  const zeroBin = Math.min(9, Math.floor(((0 - minPre) / range) * 10))

  return { active, dead, maxPre, minPre, totalEnergy, bins, zeroBin, range }
})

// ─── Step 4-5: Output computation ───
const outputComputation = computed(() => {
  if (!props.result?.hidden || !props.result?.matmul2) return []
  const matmul2 = props.result.matmul2
  const preSoftmax = props.result.pre_softmax
  const bias2 = preSoftmax.map((v, i) => v - matmul2[i])

  return Array.from({ length: 10 }, (_, d) => ({
    digit: d,
    matmul2Value: matmul2[d],
    bias2Value: bias2[d],
    preSoftmaxValue: preSoftmax[d],
    logsoftmaxValue: props.result.output[d],
    probability: Math.exp(props.result.output[d]),
  }))
})

// Step 4: Top hidden neurons contributing to the predicted digit
const topHiddenForPredicted = computed(() => {
  if (!props.result?.hidden || !props.result?.matmul2) return []
  // We'd need W2 weights for this - approximate by showing which hidden neurons are most active
  const hidden = props.result.hidden
  const entries = []
  for (let i = 0; i < HIDDEN_SIZE; i++) {
    if (hidden[i] > 0) {
      entries.push({ index: i, value: hidden[i] })
    }
  }
  entries.sort((a, b) => b.value - a.value)
  return entries.slice(0, 5)
})

// Step 5: Rank change from bias
const rankChangeFromBias = computed(() => {
  if (!outputComputation.value.length) return []
  const before = [...outputComputation.value].sort((a, b) => b.matmul2Value - a.matmul2Value)
  const after = [...outputComputation.value].sort((a, b) => b.preSoftmaxValue - a.preSoftmaxValue)
  return after.map((item, newRank) => {
    const oldRank = before.findIndex(b => b.digit === item.digit)
    return { ...item, oldRank: oldRank + 1, newRank: newRank + 1, change: oldRank - newRank }
  })
})

// ─── Step 6: LogSoftmax ───
const logsoftmaxComputation = computed(() => {
  if (!props.result?.pre_softmax) return null
  const preSoftmax = props.result.pre_softmax
  const maxVal = Math.max(...preSoftmax)
  const expValues = preSoftmax.map(v => Math.exp(v - maxVal))
  const sumExp = expValues.reduce((a, b) => a + b, 0)
  const logSumExp = Math.log(sumExp)

  return {
    maxVal, sumExp, logSumExp,
    steps: preSoftmax.map((v, i) => ({
      digit: i, raw: v, centered: v - maxVal,
      exp: expValues[i], logsoftmax: v - maxVal - logSumExp,
      probability: expValues[i] / sumExp,
    })),
  }
})

// ─── Step 7: Confidence interpretation ───
const confidenceLevel = computed(() => {
  if (!props.result) return null
  const c = props.result.confidence
  const sorted = [...outputComputation.value].sort((a, b) => b.probability - a.probability)
  const gap = sorted.length >= 2 ? sorted[0].probability - sorted[1].probability : 0
  const runner = sorted.length >= 2 ? sorted[1] : null

  let interpretation = ''
  if (c > 0.99) interpretation = '极高置信度 — 网络几乎完全确信'
  else if (c > 0.9) interpretation = '高置信度 — 非常有把握'
  else if (c > 0.7) interpretation = '中等置信度 — 有一定把握，但存在竞争'
  else if (c > 0.5) interpretation = '低置信度 — 不太确定，多个候选接近'
  else interpretation = '非常不确定 — 基本在猜'

  return { confidence: c, gap, runner, interpretation }
})

function formatNum(v) {
  if (Math.abs(v) < 0.001) return '0.0000'
  if (Math.abs(v) > 9999) return v.toExponential(2)
  return v.toFixed(4)
}

function formatShort(v) {
  if (Math.abs(v) < 0.01) return '0.00'
  if (Math.abs(v) > 999) return v.toExponential(1)
  return v.toFixed(2)
}

function pixelCoords(idx) {
  return { row: Math.floor(idx / 28), col: idx % 28 }
}
</script>

<template>
  <div class="computation-trace">
    <!-- ════════ Step 0: Input ════════ -->
    <div v-if="step === 0" class="trace-section">
      <h3 class="trace-title">输入向量：28×28 像素 → 784 维向量</h3>
      <p class="trace-desc">
        canvas 灰度值 (0-255) 经归一化映射到 [-1, 1]，然后展平为一维向量 x[784]
      </p>

      <div class="insight-box">
        <span class="insight-icon">💡</span>
        <span class="insight-text">归一化让所有输入在相同的数值范围内，防止大数值主导训练。[-1, 1] 的对称范围让正负权重同等重要。</span>
      </div>

      <!-- Concrete normalization examples -->
      <div v-if="normExamples.length" class="norm-examples">
        <h4 class="subsection-title">归一化示例（实际像素）</h4>
        <div class="formula-box">
          <div class="formula-line">
            <span class="formula-label">公式</span>
            <span class="formula-eq">=</span>
            <span class="formula-expr">(灰度值 / 255 - 0.5) / 0.5 = 灰度值 / 127.5 - 1</span>
          </div>
        </div>
        <div class="norm-example-grid">
          <div v-for="ex in normExamples" :key="ex.index" class="norm-example-row">
            <span class="norm-label" :class="ex.label === '笔画' ? 'stroke' : ex.label === '边缘' ? 'edge' : 'bg'">{{ ex.label }}</span>
            <span class="norm-coord">({{ ex.row }},{{ ex.col }})</span>
            <span class="norm-formula">
              灰度 <strong>{{ ex.gray }}</strong>
              → ({{ ex.gray }}/255 - 0.5)/0.5
              = <strong class="norm-result">{{ ex.value.toFixed(4) }}</strong>
            </span>
          </div>
        </div>
      </div>

      <!-- Stats -->
      <div v-if="inputStats" class="input-stats-grid">
        <div class="input-stat-card">
          <span class="input-stat-label">总像素</span>
          <span class="input-stat-value">{{ inputStats.total }}</span>
          <span class="input-stat-detail">28 × 28</span>
        </div>
        <div class="input-stat-card">
          <span class="input-stat-label">笔画像素</span>
          <span class="input-stat-value">{{ inputStats.brightCount }}</span>
          <span class="input-stat-detail">值 > 0</span>
        </div>
        <div class="input-stat-card">
          <span class="input-stat-label">背景占比</span>
          <span class="input-stat-value">{{ ((1 - inputStats.activeCount / inputStats.total) * 100).toFixed(0) }}%</span>
          <span class="input-stat-detail">大部分是空白</span>
        </div>
      </div>

      <!-- 784 维输入向量 -->
      <div class="input-vector-section">
        <h4 class="subsection-title">784 维输入向量 x</h4>
        <div class="input-vector-array">
          <span class="vec-bracket">[</span>
          <span v-for="(p, i) in inputVectorRows" :key="p.index" class="vec-item" :class="{ positive: p.value > 0, negative: p.value <= 0 }">
            {{ formatNum(p.value) }}<span v-if="i < inputVectorRows.length - 1" class="vec-comma">, </span>
          </span>
          <span class="vec-bracket">]</span>
        </div>
      </div>
    </div>

    <!-- ════════ Step 1: MatMul W1 ════════ -->
    <div v-if="step === 1" class="trace-section">
      <h3 class="trace-title">矩阵乘法：输入 × W1 → 隐藏层</h3>
      <p class="trace-desc">h1[j] = Σᵢ x[i] × W1[i][j]，共 784×256 = 200,704 次乘法</p>

      <div class="insight-box">
        <span class="insight-icon">💡</span>
        <span class="insight-text">每个隐藏神经元相当于一个"模式检测器"。它的权重定义了一个模板，与输入的点积越大，说明输入越匹配这个模式。</span>
      </div>

      <div class="top-neurons">
        <h4 class="subsection-title">响应最强的 8 个神经元（点击查看详情）</h4>
        <div class="neuron-cards">
          <div v-for="(n, i) in topContributorNeurons" :key="n.index"
            :class="['neuron-card', { active: activeNeuron === i }]"
            @click="activeNeuron = i">
            <span class="neuron-id">H[{{ n.index }}]</span>
            <span class="neuron-val">{{ formatNum(n.matmul1Value) }}</span>
          </div>
        </div>
      </div>

      <div v-if="selectedNeuron" class="neuron-detail">
        <h4 class="subsection-title">神经元 H[{{ selectedNeuron.index }}] 的计算过程</h4>
        <div class="formula-box">
          <div class="formula-line">
            <span class="formula-label">h1[{{ selectedNeuron.index }}]</span>
            <span class="formula-eq">=</span>
            <span class="formula-expr">
              x[0]×W1[0][{{ selectedNeuron.index }}] + x[1]×W1[1][{{ selectedNeuron.index }}] + ... + x[783]×W1[783][{{ selectedNeuron.index }}]
            </span>
          </div>
          <div class="formula-line result">
            <span class="formula-label">结果</span>
            <span class="formula-eq">=</span>
            <span class="formula-val">{{ formatNum(selectedNeuron.matmul1Value) }}</span>
            <span class="formula-note">（784 项求和）</span>
          </div>
        </div>

        <div class="pixel-contributions">
          <h5 class="sub-subsection-title">贡献最大的 12 个像素 × 权重</h5>
          <div v-if="isLoadingWeights" class="loading-hint">加载权重中...</div>
          <div v-else-if="pixelContributions.length" class="pixel-contrib-grid">
            <div v-for="pc in pixelContributions" :key="pc.index" class="pixel-contrib-row">
              <span class="pixel-coord">({{ pixelCoords(pc.index).row }},{{ pixelCoords(pc.index).col }})</span>
              <span class="pixel-val">x={{ formatShort(pc.pixel) }}</span>
              <span class="contrib-op">×</span>
              <span class="weight-val">W={{ formatShort(pc.weight) }}</span>
              <span class="contrib-op">=</span>
              <span class="contrib-val" :class="{ positive: pc.contrib > 0 }">{{ formatShort(pc.contrib) }}</span>
              <span class="contrib-bar-container">
                <span class="contrib-bar" :class="{ positive: pc.contrib > 0 }"
                  :style="{ width: (pc.absContrib / pixelContributions[0].absContrib * 100) + '%' }"></span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ════════ Step 2: Bias 1 ════════ -->
    <div v-if="step === 2" class="trace-section">
      <h3 class="trace-title">加偏置：h1 + b1 → h1b</h3>
      <p class="trace-desc">每个神经元加上对应的偏置值，共 256 个独立的加法</p>

      <div class="insight-box">
        <span class="insight-icon">💡</span>
        <span class="insight-text">
          偏置 = 神经元的"默认倾向"。正偏置让神经元更容易激活（倾向于说"是"），负偏置让它更难激活（倾向于说"不"）。
          训练会自动调整每个偏置，找到最优的激活阈值。
        </span>
      </div>

      <div v-if="biasStats" class="bias-stats">
        <span class="stat-item">偏置范围: <strong>{{ formatShort(biasStats.minBias) }} ~ {{ formatShort(biasStats.maxBias) }}</strong></span>
        <span class="stat-item">符号翻转: <strong>{{ biasStats.signFlips }}</strong> 个神经元</span>
      </div>

      <div class="bias-computation">
        <div class="bias-table-header">
          <span class="bth-neuron">神经元</span>
          <span class="bth-val">matmul</span>
          <span class="bth-op"></span>
          <span class="bth-val">偏置 b1</span>
          <span class="bth-op"></span>
          <span class="bth-val">结果</span>
          <span class="bth-effect">效果</span>
        </div>
        <div v-for="n in topContributorNeurons" :key="n.index" class="bias-row">
          <span class="bias-neuron">H[{{ n.index }}]</span>
          <span class="bias-val">{{ formatNum(n.matmul1Value) }}</span>
          <span class="bias-op">+</span>
          <span class="bias-val" :class="{ 'positive-text': n.bias1Value > 0, 'negative-text': n.bias1Value < 0 }">
            {{ formatNum(n.bias1Value) }}
          </span>
          <span class="bias-op">=</span>
          <span class="bias-result">{{ formatNum(n.preReluValue) }}</span>
          <span class="bias-effect">
            <span v-if="(n.matmul1Value > 0) !== (n.preReluValue > 0)" class="effect-flip">符号翻转!</span>
            <span v-else-if="n.bias1Value > 0" class="effect-boost">↑ 增强</span>
            <span v-else class="effect-suppress">↓ 抑制</span>
          </span>
        </div>
      </div>
    </div>

    <!-- ════════ Step 3: ReLU ════════ -->
    <div v-if="step === 3" class="trace-section">
      <h3 class="trace-title">ReLU 激活：max(0, x)</h3>
      <p class="trace-desc">负值归零，正值保留 — 这是整个网络的"开关"</p>

      <div class="insight-box">
        <span class="insight-icon">💡</span>
        <span class="insight-text">
          没有 ReLU，多层矩阵乘法等价于一层（线性函数的叠加还是线性）。ReLU 引入非线性，让网络能表达"如果看到某个模式就激活，否则忽略"的逻辑。
          被归零的神经元 = 该模式在这张图中不存在。
        </span>
      </div>

      <!-- Full stats -->
      <div v-if="reluFullStats" class="relu-overview">
        <div class="relu-stat-cards">
          <div class="relu-stat-card active-card">
            <span class="relu-stat-num">{{ reluFullStats.active }}</span>
            <span class="relu-stat-label">激活</span>
          </div>
          <div class="relu-stat-card dead-card">
            <span class="relu-stat-num">{{ reluFullStats.dead }}</span>
            <span class="relu-stat-label">归零</span>
          </div>
          <div class="relu-stat-card">
            <span class="relu-stat-num">{{ ((reluFullStats.active / 256) * 100).toFixed(0) }}%</span>
            <span class="relu-stat-label">激活率</span>
          </div>
        </div>

        <!-- Mini distribution bar -->
        <div class="relu-distribution">
          <h4 class="sub-subsection-title">256 个神经元的值分布（ReLU 前）</h4>
          <div class="dist-chart">
            <div v-for="(count, i) in reluFullStats.bins" :key="i"
              :class="['dist-bin', { 'above-zero': i >= reluFullStats.zeroBin }]"
              :style="{ height: (count / Math.max(...reluFullStats.bins)) * 80 + 20 + '%' }"
              :title="`${count} 个神经元`">
              <span class="dist-count">{{ count }}</span>
            </div>
          </div>
          <div class="dist-labels">
            <span>{{ formatShort(reluFullStats.minPre) }}</span>
            <span class="dist-zero-label">← 0 (ReLU 切割线) →</span>
            <span>{{ formatShort(reluFullStats.maxPre) }}</span>
          </div>
        </div>
      </div>

      <!-- Per-neuron details -->
      <h4 class="subsection-title" style="margin-top: 16px">Top 8 神经元 ReLU 前后对比</h4>
      <div class="relu-computation">
        <div v-for="n in topContributorNeurons" :key="n.index"
          :class="['relu-row', { activated: n.reluActivated }]">
          <span class="relu-neuron">H[{{ n.index }}]</span>
          <span class="relu-formula">
            <span class="relu-label">max(0,</span>
            <span class="relu-val" :class="{ 'negative-text': n.preReluValue < 0 }">{{ formatNum(n.preReluValue) }}</span>
            <span class="relu-label">)</span>
            <span class="relu-op">=</span>
            <span class="relu-result" :class="{ zero: !n.reluActivated }">{{ formatNum(n.hiddenValue) }}</span>
          </span>
          <span :class="['relu-status', { active: n.reluActivated }]">
            {{ n.reluActivated ? '✓ 激活' : '✗ 归零' }}
          </span>
        </div>
      </div>
    </div>

    <!-- ════════ Step 4: MatMul W2 ════════ -->
    <div v-if="step === 4" class="trace-section">
      <h3 class="trace-title">矩阵乘法：隐藏层 × W2 → 输出</h3>
      <p class="trace-desc">h2[d] = Σⱼ hidden[j] × W2[j][d]，共 256×10 = 2,560 次乘法</p>

      <div class="insight-box">
        <span class="insight-icon">💡</span>
        <span class="insight-text">
          这一步把 256 个"模式检测器"的结果汇总为 10 个数字的"得分"。
          得分高 = 多个激活的隐藏神经元都在给这个数字"投票"。
        </span>
      </div>

      <div class="output-computation">
        <div v-for="oc in outputComputation" :key="oc.digit"
          :class="['output-row', { highlight: oc.digit === result.predicted }]">
          <span class="output-digit">{{ oc.digit }}</span>
          <span class="output-val">{{ formatNum(oc.matmul2Value) }}</span>
          <span class="output-bar-container">
            <span class="output-bar"
              :class="{ positive: oc.matmul2Value > 0, negative: oc.matmul2Value <= 0 }"
              :style="{ width: Math.min(100, Math.abs(oc.matmul2Value) / Math.max(...outputComputation.map(x => Math.abs(x.matmul2Value))) * 100) + '%' }">
            </span>
          </span>
        </div>
      </div>

      <div class="hidden-summary">
        <h4 class="subsection-title">隐藏层概况</h4>
        <div class="hidden-stats">
          <span class="stat-item">激活神经元: <strong>{{ result.hidden.filter(v => v > 0).length }}</strong> / 256</span>
          <span class="stat-item">总能量: <strong>{{ formatNum(result.hidden.reduce((a, b) => a + b, 0)) }}</strong></span>
        </div>
        <div v-if="topHiddenForPredicted.length" class="top-hidden-section">
          <h5 class="sub-subsection-title">最活跃的 5 个隐藏神经元</h5>
          <div class="top-hidden-list">
            <div v-for="h in topHiddenForPredicted" :key="h.index" class="top-hidden-item">
              <span class="top-hidden-id">H[{{ h.index }}]</span>
              <span class="top-hidden-val">{{ formatNum(h.value) }}</span>
              <span class="top-hidden-bar-container">
                <span class="top-hidden-bar"
                  :style="{ width: (h.value / topHiddenForPredicted[0].value * 100) + '%' }"></span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ════════ Step 5: Bias 2 ════════ -->
    <div v-if="step === 5" class="trace-section">
      <h3 class="trace-title">加偏置：h2 + b2 → h2b</h3>
      <p class="trace-desc">10 个输出类别各自加上偏置</p>

      <div class="insight-box">
        <span class="insight-icon">💡</span>
        <span class="insight-text">
          输出偏置反映数据集中各数字出现的频率差异。如果训练集中 "1" 比 "8" 多，
          "1" 的偏置会稍大——即使没有任何输入，网络也会微微偏向更常见的数字。
        </span>
      </div>

      <div class="bias2-computation">
        <div class="bias-table-header">
          <span class="bth-neuron">数字</span>
          <span class="bth-val">matmul</span>
          <span class="bth-op"></span>
          <span class="bth-val">偏置 b2</span>
          <span class="bth-op"></span>
          <span class="bth-val">最终分数</span>
        </div>
        <div v-for="oc in outputComputation" :key="oc.digit"
          :class="['bias2-row', { highlight: oc.digit === result.predicted }]">
          <span class="bias2-digit">{{ oc.digit }}</span>
          <span class="bias2-val">{{ formatNum(oc.matmul2Value) }}</span>
          <span class="bias2-op">+</span>
          <span class="bias2-val" :class="{ 'positive-text': oc.bias2Value > 0, 'negative-text': oc.bias2Value < 0 }">
            {{ formatNum(oc.bias2Value) }}
          </span>
          <span class="bias2-op">=</span>
          <span class="bias2-result" :class="{ 'is-max': oc.digit === result.predicted }">
            {{ formatNum(oc.preSoftmaxValue) }}
          </span>
        </div>
      </div>

      <!-- Rank change -->
      <div v-if="rankChangeFromBias.some(r => r.change !== 0)" class="rank-changes">
        <h4 class="sub-subsection-title">偏置对排名的影响</h4>
        <div class="rank-change-list">
          <span v-for="r in rankChangeFromBias.filter(x => x.change !== 0)" :key="r.digit"
            class="rank-change-tag" :class="{ up: r.change > 0, down: r.change < 0 }">
            {{ r.digit }}: #{{ r.oldRank }}→#{{ r.newRank }}
            {{ r.change > 0 ? '↑' : '↓' }}
          </span>
        </div>
      </div>
    </div>

    <!-- ════════ Step 6: LogSoftmax ════════ -->
    <div v-if="step === 6" class="trace-section">
      <h3 class="trace-title">LogSoftmax：分数 → 对数概率</h3>
      <p class="trace-desc">log_softmax(x)[i] = x[i] - max(x) - log(Σ exp(x[j] - max(x)))</p>

      <div class="insight-box">
        <span class="insight-icon">💡</span>
        <span class="insight-text">
          Softmax 把任意实数分数"挤压"成概率分布（0~1，且和为 1）。取 log 是为了训练时配合 NLLLoss 更稳定。
          减去 max(x) 不改变结果，但防止 exp 溢出。
        </span>
      </div>

      <div v-if="logsoftmaxComputation" class="logsoftmax-computation">
        <div class="logsoftmax-constants">
          <span class="const-item">max(x) = <strong>{{ formatNum(logsoftmaxComputation.maxVal) }}</strong></span>
          <span class="const-item">Σ exp = <strong>{{ formatNum(logsoftmaxComputation.sumExp) }}</strong></span>
          <span class="const-item">log(Σ exp) = <strong>{{ formatNum(logsoftmaxComputation.logSumExp) }}</strong></span>
        </div>

        <div class="logsoftmax-table">
          <div class="table-header">
            <span class="th">数字</span>
            <span class="th">x[i]</span>
            <span class="th">x[i]-max</span>
            <span class="th">exp</span>
            <span class="th">log_softmax</span>
            <span class="th">概率</span>
          </div>
          <div v-for="s in logsoftmaxComputation.steps" :key="s.digit"
            class="table-row"
            :class="{ top: s.digit === result.predicted }">
            <span class="td digit">{{ s.digit }}</span>
            <span class="td">{{ formatNum(s.raw) }}</span>
            <span class="td">{{ formatNum(s.centered) }}</span>
            <span class="td">{{ formatNum(s.exp) }}</span>
            <span class="td">{{ formatNum(s.logsoftmax) }}</span>
            <span class="td prob">{{ (s.probability * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- ════════ Step 7: Argmax ════════ -->
    <div v-if="step === 7" class="trace-section">
      <h3 class="trace-title">Argmax：取最大概率对应的数字</h3>
      <p class="trace-desc">遍历 10 个输出值，找到概率最大的索引</p>

      <div class="argmax-display">
        <div class="argmax-winner">
          <span class="winner-label">预测结果</span>
          <span class="winner-number">{{ result.predicted }}</span>
          <span class="winner-conf">置信度: {{ (result.confidence * 100).toFixed(1) }}%</span>
        </div>

        <!-- Confidence interpretation -->
        <div v-if="confidenceLevel" class="confidence-interpretation">
          <div class="conf-insight" :class="{
            'conf-high': confidenceLevel.confidence > 0.9,
            'conf-mid': confidenceLevel.confidence > 0.5 && confidenceLevel.confidence <= 0.9,
            'conf-low': confidenceLevel.confidence <= 0.5,
          }">
            <span class="conf-insight-text">{{ confidenceLevel.interpretation }}</span>
          </div>
          <div v-if="confidenceLevel.runner" class="conf-detail">
            <span>最大竞争者: <strong>{{ confidenceLevel.runner.digit }}</strong>（{{ (confidenceLevel.runner.probability * 100).toFixed(1) }}%）</span>
            <span>概率差距: <strong>{{ (confidenceLevel.gap * 100).toFixed(1) }}%</strong></span>
          </div>
        </div>

        <div class="argmax-ranks">
          <div v-for="(oc, i) in [...outputComputation].sort((a, b) => b.probability - a.probability)"
            :key="oc.digit"
            :class="['rank-row', { winner: oc.digit === result.predicted }]">
            <span class="rank-num">#{{ i + 1 }}</span>
            <span class="rank-digit">{{ oc.digit }}</span>
            <span class="rank-prob">{{ (oc.probability * 100).toFixed(2) }}%</span>
            <span class="rank-bar-container">
              <span class="rank-bar"
                :style="{ width: (oc.probability / outputComputation.reduce((m, x) => Math.max(m, x.probability), 0) * 100) + '%' }">
              </span>
            </span>
          </div>
        </div>

        <div class="insight-box" style="margin-top: 16px">
          <span class="insight-icon">💡</span>
          <span class="insight-text">
            整个前向传播只用了加法和乘法（以及 max、exp、log），没有任何"如果是数字 3 就..."的规则。
            网络通过训练自动学会了从像素模式中提取特征并分类。
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.computation-trace { width: 100%; }

.trace-section {
  background: var(--color-card);
  border-radius: 14px;
  padding: 20px;
  box-shadow: var(--shadow-sm);
}

.trace-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; color: var(--color-text); }
.trace-desc { font-size: 13px; color: var(--color-text-light); margin-bottom: 16px; }
.subsection-title { font-size: 14px; font-weight: 600; margin-bottom: 10px; color: var(--color-text); }
.sub-subsection-title { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--color-text-light); }

/* ─── Insight callout ─── */
.insight-box {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 12px 14px;
  background: linear-gradient(135deg, rgba(108, 99, 255, 0.06), rgba(0, 210, 255, 0.06));
  border-left: 3px solid var(--color-primary);
  border-radius: 0 10px 10px 0;
  margin-bottom: 16px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text);
}
.insight-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
.insight-text { flex: 1; }

/* ─── Step 0: Input ─── */
.norm-examples { margin-bottom: 16px; }
.norm-example-grid { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.norm-example-row {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 12px; background: var(--color-bg); border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
}
.norm-label {
  font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; min-width: 36px; text-align: center;
}
.norm-label.stroke { background: rgba(108, 99, 255, 0.15); color: var(--color-primary); }
.norm-label.edge { background: rgba(255, 193, 7, 0.15); color: #FFC107; }
.norm-label.bg { background: rgba(255, 82, 82, 0.1); color: var(--color-danger); }
.norm-coord { color: var(--color-text-light); min-width: 40px; }
.norm-formula { color: var(--color-text-light); }
.norm-result { color: var(--color-primary); }

.input-stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px; }
.input-stat-card {
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  padding: 10px; background: var(--color-bg); border-radius: 10px;
}
.input-stat-label { font-size: 12px; color: var(--color-text-light); font-weight: 600; }
.input-stat-value { font-size: 18px; font-weight: 800; color: var(--color-primary); font-family: 'SF Mono', 'Fira Code', monospace; }
.input-stat-detail { font-size: 11px; color: var(--color-text-light); }

.input-brightest { margin-bottom: 16px; }
.bright-pixel-grid { display: flex; flex-direction: column; gap: 4px; }
.bright-pixel-row {
  display: flex; align-items: center; gap: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
  padding: 4px 8px; background: var(--color-bg); border-radius: 6px;
}
.pixel-idx { color: var(--color-text-light); min-width: 55px; }
.pixel-value { color: var(--color-accent); font-weight: 700; min-width: 55px; }
.pixel-bar-container { flex: 1; height: 10px; background: var(--color-border); border-radius: 5px; overflow: hidden; }
.pixel-bar { height: 100%; background: linear-gradient(90deg, var(--color-accent), var(--color-primary)); border-radius: 5px; }

.input-vector-section { margin-bottom: 16px; }
.input-vector-array {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.8;
  background: #1e1e2e;
  border-radius: 10px;
  padding: 16px;
  max-height: 300px;
  overflow-y: auto;
  word-break: break-all;
}
.vec-bracket { font-size: 16px; font-weight: 700; color: #89b4fa; }
.vec-item { color: #cdd6f4; }
.vec-item.positive { color: #a6e3a1; }
.vec-item.negative { color: #6c7086; }
.vec-comma { color: #585b70; }

/* ─── Step 1: MatMul ─── */
.top-neurons { margin-bottom: 16px; }
.neuron-cards { display: flex; flex-wrap: wrap; gap: 8px; }
.neuron-card {
  padding: 8px 14px; border-radius: 8px; background: var(--color-bg);
  cursor: pointer; transition: var(--transition);
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  border: 2px solid transparent;
}
.neuron-card:hover { background: var(--color-border); }
.neuron-card.active { border-color: var(--color-primary); background: rgba(108, 99, 255, 0.08); }
.neuron-id { font-size: 12px; font-weight: 700; color: var(--color-text-light); font-family: 'SF Mono', 'Fira Code', monospace; }
.neuron-val { font-size: 14px; font-weight: 800; color: var(--color-primary); font-family: 'SF Mono', 'Fira Code', monospace; }

.neuron-detail { margin-top: 16px; }
.formula-box { background: #1e1e2e; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
.formula-line {
  display: flex; align-items: center; gap: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
  color: #cdd6f4; margin-bottom: 6px;
}
.formula-line.result { margin-top: 8px; padding-top: 8px; border-top: 1px solid #313244; }
.formula-label { color: #89b4fa; min-width: 60px; }
.formula-eq { color: #f9e2af; }
.formula-expr { color: #a6e3a1; word-break: break-all; }
.formula-val { color: #f38ba8; font-weight: 700; font-size: 15px; }
.formula-note { color: #6c7086; font-size: 12px; }

.pixel-contributions { margin-top: 12px; }
.loading-hint { font-size: 13px; color: var(--color-text-light); padding: 8px; }
.pixel-contrib-grid { display: flex; flex-direction: column; gap: 4px; }
.pixel-contrib-row {
  display: flex; align-items: center; gap: 6px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
  padding: 4px 8px; background: var(--color-bg); border-radius: 6px;
}
.pixel-coord { color: var(--color-text-light); min-width: 40px; }
.pixel-val, .weight-val { color: var(--color-text); min-width: 60px; }
.contrib-op { color: var(--color-primary); font-weight: 700; }
.contrib-val { font-weight: 700; color: var(--color-danger); min-width: 50px; }
.contrib-val.positive { color: var(--color-success); }
.contrib-bar-container { flex: 1; height: 8px; background: var(--color-border); border-radius: 4px; overflow: hidden; }
.contrib-bar { height: 100%; border-radius: 4px; background: var(--color-danger); }
.contrib-bar.positive { background: var(--color-success); }

/* ─── Step 2 & 5: Bias ─── */
.bias-stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.bias-stats .stat-item, .hidden-stats .stat-item {
  font-size: 13px; color: var(--color-text-light);
  background: var(--color-bg); padding: 6px 12px; border-radius: 8px;
}
.bias-stats .stat-item strong, .hidden-stats .stat-item strong { color: var(--color-text); }

.bias-table-header {
  display: flex; align-items: center; gap: 12px;
  padding: 6px 12px; font-size: 11px; font-weight: 700;
  color: var(--color-text-light); text-transform: uppercase; letter-spacing: 0.5px;
}
.bth-neuron { font-weight: 700; min-width: 80px; }
.bth-val { min-width: 70px; text-align: right; }
.bth-op { width: 16px; text-align: center; }
.bth-effect { flex: 1; text-align: right; }

.bias-computation, .bias2-computation { display: flex; flex-direction: column; gap: 4px; }
.bias-row, .bias2-row {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 12px; background: var(--color-bg); border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
  transition: var(--transition);
}
.bias-row.highlight, .bias2-row.highlight { background: rgba(108, 99, 255, 0.08); }
.bias-neuron, .bias2-digit { font-weight: 700; color: var(--color-text); min-width: 80px; }
.bias-val, .bias2-val { color: var(--color-text-light); min-width: 70px; text-align: right; }
.bias-op, .bias2-op { color: var(--color-primary); font-weight: 700; width: 16px; text-align: center; }
.bias-result, .bias2-result { color: var(--color-primary); font-weight: 700; min-width: 70px; text-align: right; }
.bias2-result.is-max { color: var(--color-success); font-weight: 800; }

.positive-text { color: var(--color-success) !important; }
.negative-text { color: var(--color-danger) !important; }

.bias-effect { flex: 1; text-align: right; font-size: 11px; }
.effect-flip { color: #FFC107; font-weight: 700; }
.effect-boost { color: var(--color-success); }
.effect-suppress { color: var(--color-danger); }

.rank-changes { margin-top: 12px; }
.rank-change-list { display: flex; gap: 8px; flex-wrap: wrap; }
.rank-change-tag {
  padding: 4px 10px; border-radius: 6px; font-size: 12px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-weight: 600;
}
.rank-change-tag.up { background: rgba(76, 175, 80, 0.1); color: var(--color-success); }
.rank-change-tag.down { background: rgba(255, 82, 82, 0.1); color: var(--color-danger); }

/* ─── Step 3: ReLU ─── */
.relu-overview { margin-bottom: 16px; }
.relu-stat-cards { display: flex; gap: 10px; margin-bottom: 16px; }
.relu-stat-card {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  padding: 12px; border-radius: 10px; background: var(--color-bg);
}
.relu-stat-card.active-card { border-left: 3px solid var(--color-success); }
.relu-stat-card.dead-card { border-left: 3px solid var(--color-danger); }
.relu-stat-num { font-size: 24px; font-weight: 800; color: var(--color-text); font-family: 'SF Mono', 'Fira Code', monospace; }
.relu-stat-label { font-size: 12px; color: var(--color-text-light); font-weight: 600; }

.relu-distribution { margin-bottom: 12px; }
.dist-chart {
  display: flex; align-items: flex-end; gap: 3px; height: 80px;
  padding: 8px 4px; background: var(--color-bg); border-radius: 10px;
}
.dist-bin {
  flex: 1; border-radius: 4px 4px 0 0; min-height: 4px;
  background: rgba(255, 82, 82, 0.4); position: relative;
  display: flex; align-items: flex-start; justify-content: center;
  transition: var(--transition);
}
.dist-bin.above-zero { background: rgba(76, 175, 80, 0.5); }
.dist-count { font-size: 9px; font-weight: 700; color: var(--color-text-light); margin-top: 2px; }
.dist-labels {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 11px; color: var(--color-text-light); margin-top: 4px;
  font-family: 'SF Mono', 'Fira Code', monospace;
}
.dist-zero-label { font-size: 10px; color: var(--color-primary); font-weight: 600; }

.relu-computation { display: flex; flex-direction: column; gap: 4px; }
.relu-row {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 12px; background: var(--color-bg); border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
}
.relu-neuron { font-weight: 700; color: var(--color-text); min-width: 80px; }
.relu-formula { display: flex; align-items: center; gap: 4px; }
.relu-label { color: var(--color-text-light); }
.relu-val { color: var(--color-text); }
.relu-op { color: var(--color-secondary); font-weight: 700; }
.relu-result { color: var(--color-success); font-weight: 700; }
.relu-result.zero { color: var(--color-text-light); }
.relu-status {
  font-size: 12px; padding: 2px 8px; border-radius: 4px;
  background: rgba(255, 82, 82, 0.1); color: var(--color-danger);
  margin-left: auto;
}
.relu-status.active { background: rgba(76, 175, 80, 0.1); color: var(--color-success); }

/* ─── Step 4: Output ─── */
.output-computation { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; }
.output-row {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 10px; background: var(--color-bg); border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
  transition: var(--transition);
}
.output-row.highlight { background: rgba(108, 99, 255, 0.08); }
.output-digit { font-weight: 800; font-size: 16px; min-width: 24px; text-align: center; }
.output-val { min-width: 70px; color: var(--color-text); }
.output-bar-container { flex: 1; height: 16px; background: var(--color-border); border-radius: 8px; overflow: hidden; }
.output-bar { height: 100%; border-radius: 8px; transition: width 0.3s ease; }
.output-bar.positive { background: linear-gradient(90deg, var(--color-primary), var(--color-accent)); }
.output-bar.negative { background: linear-gradient(90deg, var(--color-secondary), #FF8E53); }

.hidden-summary { margin-top: 12px; }
.hidden-stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }

.top-hidden-section { margin-top: 8px; }
.top-hidden-list { display: flex; flex-direction: column; gap: 4px; }
.top-hidden-item {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 10px; background: var(--color-bg); border-radius: 6px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
}
.top-hidden-id { color: var(--color-text-light); min-width: 50px; }
.top-hidden-val { color: var(--color-primary); font-weight: 700; min-width: 60px; }
.top-hidden-bar-container { flex: 1; height: 8px; background: var(--color-border); border-radius: 4px; overflow: hidden; }
.top-hidden-bar { height: 100%; background: linear-gradient(90deg, var(--color-primary), var(--color-accent)); border-radius: 4px; }

/* ─── Step 6: LogSoftmax ─── */
.logsoftmax-computation { margin-top: 12px; }
.logsoftmax-constants {
  display: flex; gap: 20px; flex-wrap: wrap;
  margin-bottom: 16px; padding: 12px;
  background: var(--color-bg); border-radius: 10px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
}
.const-item { color: var(--color-text-light); }
.const-item strong { color: var(--color-primary); }
.logsoftmax-table { border-radius: 10px; overflow: hidden; border: 1px solid var(--color-border); }
.table-header {
  display: grid; grid-template-columns: 50px repeat(5, 1fr);
  background: var(--color-bg); font-size: 12px; font-weight: 700; color: var(--color-text-light);
}
.table-header .th { padding: 8px 10px; text-align: center; }
.table-row {
  display: grid; grid-template-columns: 50px repeat(5, 1fr);
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
  border-top: 1px solid var(--color-border);
}
.table-row .td { padding: 8px 10px; text-align: center; color: var(--color-text); }
.table-row .td.digit { font-weight: 800; font-size: 15px; }
.table-row .td.prob { font-weight: 700; color: var(--color-primary); }
.table-row.top { background: rgba(108, 99, 255, 0.06); }

/* ─── Step 7: Argmax ─── */
.argmax-display { margin-top: 12px; }
.argmax-winner {
  display: flex; flex-direction: column; align-items: center; gap: 8px;
  margin-bottom: 16px; padding: 20px;
  background: linear-gradient(135deg, rgba(108, 99, 255, 0.08), rgba(0, 210, 255, 0.08));
  border-radius: 14px;
}
.winner-label { font-size: 14px; color: var(--color-text-light); font-weight: 600; }
.winner-number {
  font-size: 56px; font-weight: 900;
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.winner-conf { font-size: 14px; color: var(--color-text-light); font-weight: 600; }

.confidence-interpretation { margin-bottom: 16px; }
.conf-insight {
  padding: 10px 14px; border-radius: 8px; font-size: 14px; font-weight: 600; text-align: center;
}
.conf-high { background: rgba(76, 175, 80, 0.1); color: var(--color-success); }
.conf-mid { background: rgba(255, 193, 7, 0.1); color: #FFC107; }
.conf-low { background: rgba(255, 82, 82, 0.1); color: var(--color-danger); }
.conf-detail {
  display: flex; gap: 20px; justify-content: center;
  margin-top: 8px; font-size: 13px; color: var(--color-text-light);
}
.conf-detail strong { color: var(--color-text); }

.argmax-ranks { display: flex; flex-direction: column; gap: 6px; }
.rank-row {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 12px; background: var(--color-bg); border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
  transition: var(--transition);
}
.rank-row.winner { background: rgba(108, 99, 255, 0.1); border: 1px solid rgba(108, 99, 255, 0.3); }
.rank-num { min-width: 30px; color: var(--color-text-light); font-weight: 600; }
.rank-digit { font-size: 18px; font-weight: 800; min-width: 28px; text-align: center; }
.rank-prob { min-width: 70px; font-weight: 700; color: var(--color-text); }
.rank-bar-container { flex: 1; height: 12px; background: var(--color-border); border-radius: 6px; overflow: hidden; }
.rank-bar {
  height: 100%; background: linear-gradient(90deg, var(--color-primary), var(--color-accent));
  border-radius: 6px; transition: width 0.5s ease;
}
.rank-row.winner .rank-bar { background: linear-gradient(90deg, var(--color-success), var(--color-accent)); }
</style>
