<script setup>
import { ref, computed, watch, onMounted } from 'vue'

const props = defineProps({
  step: { type: Number, default: -1 },
  result: { type: Object, default: null },
})

const activeNeuron = ref(0)
const topContributors = ref([])
const isLoading = ref(false)

const HIDDEN_SIZE = 256

const topContributorCount = computed(() => {
  if (!props.result?.pixels || !props.result?.matmul1) return 0
  return Math.min(8, HIDDEN_SIZE)
})

const topContributorNeurons = computed(() => {
  if (!props.result?.pixels || !props.result?.matmul1) return []
  const pixels = props.result.pixels
  const matmul1 = props.result.matmul1

  const neurons = []
  for (let h = 0; h < HIDDEN_SIZE; h++) {
    neurons.push({
      index: h,
      value: matmul1[h],
      absValue: Math.abs(matmul1[h]),
    })
  }
  neurons.sort((a, b) => b.absValue - a.absValue)
  return neurons.slice(0, topContributorCount.value)
})

const selectedNeuronData = computed(() => {
  if (!props.result || !topContributorNeurons.value.length) return null

  const neuronIdx = topContributorNeurons.value[activeNeuron.value]?.index ?? 0
  const pixels = props.result.pixels
  const matmul1 = props.result.matmul1
  const preRelu = props.result.pre_relu
  const hidden = props.result.hidden
  const matmul2 = props.result.matmul2
  const preSoftmax = props.result.pre_softmax
  const output = props.result.output

  // Find top contributing pixels for this neuron
  const pixelContributions = []
  for (let i = 0; i < 784; i++) {
    pixelContributions.push({
      index: i,
      pixel: pixels[i],
    })
  }

  return {
    neuronIdx,
    matmul1Value: matmul1[neuronIdx],
    bias1Value: preRelu[neuronIdx] - matmul1[neuronIdx],
    preReluValue: preRelu[neuronIdx],
    hiddenValue: hidden[neuronIdx],
    reluActivated: hidden[neuronIdx] > 0,
    matmul2Values: matmul2,
    preSoftmaxValues: preSoftmax,
    outputValues: output,
    pixelContributions,
  }
})

const outputComputation = computed(() => {
  if (!props.result?.hidden || !props.result?.matmul2) return []
  const hidden = props.result.hidden
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

const logsoftmaxComputation = computed(() => {
  if (!props.result?.pre_softmax) return null
  const preSoftmax = props.result.pre_softmax
  const maxVal = Math.max(...preSoftmax)
  const expValues = preSoftmax.map(v => Math.exp(v - maxVal))
  const sumExp = expValues.reduce((a, b) => a + b, 0)
  const logSumExp = Math.log(sumExp)

  return {
    maxVal,
    sumExp,
    logSumExp,
    steps: preSoftmax.map((v, i) => ({
      digit: i,
      raw: v,
      centered: v - maxVal,
      exp: expValues[i],
      logsoftmax: v - maxVal - logSumExp,
      probability: expValues[i] / sumExp,
    })),
  }
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
    <!-- Step 1: Matrix Multiplication (Input → Hidden) -->
    <div v-if="step === 1" class="trace-section">
      <h3 class="trace-title">矩阵乘法：输入 × W1 → 隐藏层</h3>
      <p class="trace-desc">
        h1[j] = Σᵢ x[i] × W1[i][j]，共 784×256 = 200,704 次乘法
      </p>

      <div class="top-neurons">
        <h4 class="subsection-title">绝对值最大的 {{ topContributorCount }} 个神经元（matmul 后，加偏置前）</h4>
        <div class="neuron-cards">
          <div
            v-for="(n, i) in topContributorNeurons"
            :key="n.index"
            :class="['neuron-card', { active: activeNeuron === i }]"
            @click="activeNeuron = i"
          >
            <span class="neuron-id">H[{{ n.index }}]</span>
            <span class="neuron-val">{{ formatNum(n.value) }}</span>
          </div>
        </div>
      </div>

      <div v-if="selectedNeuronData" class="neuron-detail">
        <h4 class="subsection-title">
          神经元 H[{{ selectedNeuronData.neuronIdx }}] 的计算过程
        </h4>
        <div class="formula-box">
          <div class="formula-line">
            <span class="formula-label">h1[{{ selectedNeuronData.neuronIdx }}]</span>
            <span class="formula-eq">=</span>
            <span class="formula-expr">
              x[0]×W1[0][{{ selectedNeuronData.neuronIdx }}] + x[1]×W1[1][{{ selectedNeuronData.neuronIdx }}] + ... + x[783]×W1[783][{{ selectedNeuronData.neuronIdx }}]
            </span>
          </div>
          <div class="formula-line result">
            <span class="formula-label">结果</span>
            <span class="formula-eq">=</span>
            <span class="formula-val">{{ formatNum(selectedNeuronData.matmul1Value) }}</span>
          </div>
        </div>

        <div class="pixel-contributions">
          <h5 class="sub-subsection-title">贡献最大的像素（|x[i] × W1[i][h]| 最大）</h5>
          <div class="pixel-contrib-grid">
            <template v-for="(pc, i) in selectedNeuronData.pixelContributions
              .map((pc, idx) => ({ ...pc, weight: pc.pixel }))
              .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
              .slice(0, 12)" :key="pc.index">
              <div class="pixel-contrib-row">
                <span class="pixel-coord">({{ pixelCoords(pc.index).row }},{{ pixelCoords(pc.index).col }})</span>
                <span class="pixel-val">x = {{ formatShort(pc.pixel) }}</span>
                <span class="contrib-op">×</span>
                <span class="weight-val">W = {{ formatShort(pc.pixel) }}</span>
                <span class="contrib-op">=</span>
                <span class="contrib-val" :class="{ positive: pc.pixel * pc.pixel > 0 }">
                  {{ formatShort(pc.pixel * pc.pixel) }}
                </span>
              </div>
            </template>
          </div>
          <p class="contrib-note">注：以上为简化展示，实际 W1 权重需从模型加载</p>
        </div>
      </div>
    </div>

    <!-- Step 2: Add Bias 1 -->
    <div v-if="step === 2" class="trace-section">
      <h3 class="trace-title">加偏置：h1 + b1 → h1b</h3>
      <p class="trace-desc">每个神经元加上对应的偏置值</p>

      <div class="bias-computation">
        <div
          v-for="n in topContributorNeurons"
          :key="n.index"
          class="bias-row"
        >
          <span class="bias-neuron">H[{{ n.index }}]</span>
          <span class="bias-formula">
            <span class="bias-val">{{ formatNum(n.value) }}</span>
            <span class="bias-op">+</span>
            <span class="bias-val">{{ formatNum(selectedNeuronData?.bias1Value ?? 0) }}</span>
            <span class="bias-op">=</span>
            <span class="bias-result">{{ formatNum(selectedNeuronData?.preReluValue ?? 0) }}</span>
          </span>
        </div>
      </div>
    </div>

    <!-- Step 3: ReLU -->
    <div v-if="step === 3" class="trace-section">
      <h3 class="trace-title">ReLU 激活：max(0, x)</h3>
      <p class="trace-desc">负值归零，正值保留</p>

      <div class="relu-computation">
        <div
          v-for="n in topContributorNeurons"
          :key="n.index"
          :class="['relu-row', { activated: selectedNeuronData?.reluActivated }]"
        >
          <span class="relu-neuron">H[{{ n.index }}]</span>
          <span class="relu-formula">
            <span class="relu-label">max(0,</span>
            <span class="relu-val">{{ formatNum(selectedNeuronData?.preReluValue ?? 0) }}</span>
            <span class="relu-label">)</span>
            <span class="relu-op">=</span>
            <span class="relu-result" :class="{ zero: !selectedNeuronData?.reluActivated }">
              {{ formatNum(selectedNeuronData?.hiddenValue ?? 0) }}
            </span>
          </span>
          <span :class="['relu-status', { active: selectedNeuronData?.reluActivated }]">
            {{ selectedNeuronData?.reluActivated ? '✓ 激活' : '✗ 归零' }}
          </span>
        </div>
      </div>
    </div>

    <!-- Step 4: Matrix Multiplication (Hidden → Output) -->
    <div v-if="step === 4" class="trace-section">
      <h3 class="trace-title">矩阵乘法：隐藏层 × W2 → 输出</h3>
      <p class="trace-desc">h2[d] = Σⱼ hidden[j] × W2[j][d]，共 256×10 = 2,560 次乘法</p>

      <div class="output-computation">
        <div v-for="oc in outputComputation" :key="oc.digit" class="output-row">
          <span class="output-digit">{{ oc.digit }}</span>
          <span class="output-formula">
            <span class="output-val">{{ formatNum(oc.matmul2Value) }}</span>
          </span>
          <span class="output-bar-container">
            <span
              class="output-bar"
              :class="{ positive: oc.matmul2Value > 0, negative: oc.matmul2Value <= 0 }"
              :style="{ width: Math.min(100, Math.abs(oc.matmul2Value) * 20) + '%' }"
            ></span>
          </span>
        </div>
      </div>

      <div class="hidden-summary">
        <h4 class="subsection-title">隐藏层激活摘要</h4>
        <div class="hidden-stats">
          <span class="stat-item">
            激活神经元: <strong>{{ result.hidden.filter(v => v > 0).length }}</strong> / 256
          </span>
          <span class="stat-item">
            最大值: <strong>{{ formatNum(Math.max(...result.hidden)) }}</strong>
          </span>
          <span class="stat-item">
            L1 范数: <strong>{{ formatNum(result.hidden.reduce((a, b) => a + Math.abs(b), 0)) }}</strong>
          </span>
        </div>
      </div>
    </div>

    <!-- Step 5: Add Bias 2 -->
    <div v-if="step === 5" class="trace-section">
      <h3 class="trace-title">加偏置：h2 + b2 → h2b</h3>
      <p class="trace-desc">10 个输出类别各自加上偏置</p>

      <div class="bias2-computation">
        <div v-for="oc in outputComputation" :key="oc.digit" class="bias2-row">
          <span class="bias2-digit">类别 {{ oc.digit }}</span>
          <span class="bias2-formula">
            <span class="bias2-val">{{ formatNum(oc.matmul2Value) }}</span>
            <span class="bias2-op">+</span>
            <span class="bias2-val">{{ formatNum(oc.bias2Value) }}</span>
            <span class="bias2-op">=</span>
            <span class="bias2-result">{{ formatNum(oc.preSoftmaxValue) }}</span>
          </span>
        </div>
      </div>
    </div>

    <!-- Step 6: LogSoftmax -->
    <div v-if="step === 6" class="trace-section">
      <h3 class="trace-title">LogSoftmax：分数 → 对数概率</h3>
      <p class="trace-desc">log_softmax(x)[i] = x[i] - max(x) - log(Σ exp(x[j] - max(x)))</p>

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
            <span class="th">exp(x[i]-max)</span>
            <span class="th">log_softmax</span>
            <span class="th">概率</span>
          </div>
          <div
            v-for="s in logsoftmaxComputation.steps"
            :key="s.digit"
            class="table-row"
            :class="{ top: s.probability === Math.max(...logsoftmaxComputation.steps.map(x => x.probability)) }"
          >
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

    <!-- Step 7: Argmax -->
    <div v-if="step === 7" class="trace-section">
      <h3 class="trace-title">Argmax：取最大概率对应的数字</h3>
      <p class="trace-desc">遍历 10 个输出值，找到概率最大的索引</p>

      <div class="argmax-display">
        <div class="argmax-winner">
          <span class="winner-label">预测结果</span>
          <span class="winner-number">{{ result.predicted }}</span>
          <span class="winner-conf">置信度: {{ (result.confidence * 100).toFixed(1) }}%</span>
        </div>

        <div class="argmax-ranks">
          <div
            v-for="(oc, i) in [...outputComputation].sort((a, b) => b.probability - a.probability)"
            :key="oc.digit"
            :class="['rank-row', { winner: oc.digit === result.predicted }]"
          >
            <span class="rank-num">#{{ i + 1 }}</span>
            <span class="rank-digit">{{ oc.digit }}</span>
            <span class="rank-prob">{{ (oc.probability * 100).toFixed(2) }}%</span>
            <span class="rank-bar-container">
              <span
                class="rank-bar"
                :style="{ width: (oc.probability / outputComputation.reduce((m, x) => Math.max(m, x.probability), 0) * 100) + '%' }"
              ></span>
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.computation-trace {
  width: 100%;
}

.trace-section {
  background: var(--color-card);
  border-radius: 14px;
  padding: 20px;
  box-shadow: var(--shadow-sm);
}

.trace-title {
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 6px;
  color: var(--color-text);
}

.trace-desc {
  font-size: 13px;
  color: var(--color-text-light);
  margin-bottom: 16px;
}

.subsection-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
  color: var(--color-text);
}

.sub-subsection-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--color-text-light);
}

/* Top neurons */
.top-neurons {
  margin-bottom: 16px;
}

.neuron-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.neuron-card {
  padding: 8px 14px;
  border-radius: 8px;
  background: var(--color-bg);
  cursor: pointer;
  transition: var(--transition);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  border: 2px solid transparent;
}

.neuron-card:hover {
  background: var(--color-border);
}

.neuron-card.active {
  border-color: var(--color-primary);
  background: rgba(108, 99, 255, 0.08);
}

.neuron-id {
  font-size: 12px;
  font-weight: 700;
  color: var(--color-text-light);
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.neuron-val {
  font-size: 14px;
  font-weight: 800;
  color: var(--color-primary);
  font-family: 'SF Mono', 'Fira Code', monospace;
}

/* Neuron detail */
.neuron-detail {
  margin-top: 16px;
}

.formula-box {
  background: #1e1e2e;
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
}

.formula-line {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: #cdd6f4;
  margin-bottom: 6px;
}

.formula-line.result {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #313244;
}

.formula-label {
  color: #89b4fa;
  min-width: 60px;
}

.formula-eq {
  color: #f9e2af;
}

.formula-expr {
  color: #a6e3a1;
  word-break: break-all;
}

.formula-val {
  color: #f38ba8;
  font-weight: 700;
  font-size: 15px;
}

/* Pixel contributions */
.pixel-contributions {
  margin-top: 12px;
}

.pixel-contrib-grid {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.pixel-contrib-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  padding: 4px 8px;
  background: var(--color-bg);
  border-radius: 6px;
}

.pixel-coord {
  color: var(--color-text-light);
  min-width: 40px;
}

.pixel-val,
.weight-val {
  color: var(--color-text);
  min-width: 50px;
}

.contrib-op {
  color: var(--color-primary);
  font-weight: 700;
}

.contrib-val {
  font-weight: 700;
}

.contrib-val.positive {
  color: var(--color-success);
}

.contrib-val:not(.positive) {
  color: var(--color-danger);
}

.contrib-note {
  font-size: 11px;
  color: var(--color-text-light);
  margin-top: 8px;
  font-style: italic;
}

/* Bias computation */
.bias-computation,
.bias2-computation {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.bias-row,
.bias2-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  background: var(--color-bg);
  border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
}

.bias-neuron,
.bias2-digit {
  font-weight: 700;
  color: var(--color-text);
  min-width: 80px;
}

.bias-formula,
.bias2-formula {
  display: flex;
  align-items: center;
  gap: 6px;
}

.bias-val,
.bias2-val {
  color: var(--color-text-light);
}

.bias-op,
.bias2-op {
  color: var(--color-primary);
  font-weight: 700;
}

.bias-result,
.bias2-result {
  color: var(--color-primary);
  font-weight: 700;
}

/* ReLU computation */
.relu-computation {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.relu-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  background: var(--color-bg);
  border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
}

.relu-neuron {
  font-weight: 700;
  color: var(--color-text);
  min-width: 80px;
}

.relu-formula {
  display: flex;
  align-items: center;
  gap: 4px;
}

.relu-label {
  color: var(--color-text-light);
}

.relu-val {
  color: var(--color-text);
}

.relu-op {
  color: var(--color-secondary);
  font-weight: 700;
}

.relu-result {
  color: var(--color-success);
  font-weight: 700;
}

.relu-result.zero {
  color: var(--color-text-light);
}

.relu-status {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 4px;
  background: rgba(255, 82, 82, 0.1);
  color: var(--color-danger);
}

.relu-status.active {
  background: rgba(76, 175, 80, 0.1);
  color: var(--color-success);
}

/* Output computation */
.output-computation {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 16px;
}

.output-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  background: var(--color-bg);
  border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
}

.output-digit {
  font-weight: 800;
  font-size: 16px;
  min-width: 24px;
  text-align: center;
}

.output-val {
  min-width: 70px;
  color: var(--color-text);
}

.output-bar-container {
  flex: 1;
  height: 16px;
  background: var(--color-border);
  border-radius: 8px;
  overflow: hidden;
}

.output-bar {
  height: 100%;
  border-radius: 8px;
  transition: width 0.3s ease;
}

.output-bar.positive {
  background: linear-gradient(90deg, var(--color-primary), var(--color-accent));
}

.output-bar.negative {
  background: linear-gradient(90deg, var(--color-secondary), #FF8E53);
}

.hidden-summary {
  margin-top: 12px;
}

.hidden-stats {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.stat-item {
  font-size: 13px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 6px 12px;
  border-radius: 8px;
}

.stat-item strong {
  color: var(--color-text);
}

/* LogSoftmax */
.logsoftmax-computation {
  margin-top: 12px;
}

.logsoftmax-constants {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 16px;
  padding: 12px;
  background: var(--color-bg);
  border-radius: 10px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
}

.const-item {
  color: var(--color-text-light);
}

.const-item strong {
  color: var(--color-primary);
}

.logsoftmax-table {
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--color-border);
}

.table-header {
  display: grid;
  grid-template-columns: 50px repeat(5, 1fr);
  background: var(--color-bg);
  font-size: 12px;
  font-weight: 700;
  color: var(--color-text-light);
}

.table-header .th {
  padding: 8px 10px;
  text-align: center;
}

.table-row {
  display: grid;
  grid-template-columns: 50px repeat(5, 1fr);
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  border-top: 1px solid var(--color-border);
}

.table-row .td {
  padding: 8px 10px;
  text-align: center;
  color: var(--color-text);
}

.table-row .td.digit {
  font-weight: 800;
  font-size: 15px;
}

.table-row .td.prob {
  font-weight: 700;
  color: var(--color-primary);
}

.table-row.top {
  background: rgba(108, 99, 255, 0.06);
}

/* Argmax */
.argmax-display {
  margin-top: 12px;
}

.argmax-winner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  margin-bottom: 20px;
  padding: 20px;
  background: linear-gradient(135deg, rgba(108, 99, 255, 0.08), rgba(0, 210, 255, 0.08));
  border-radius: 14px;
}

.winner-label {
  font-size: 14px;
  color: var(--color-text-light);
  font-weight: 600;
}

.winner-number {
  font-size: 56px;
  font-weight: 900;
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.winner-conf {
  font-size: 14px;
  color: var(--color-text-light);
  font-weight: 600;
}

.argmax-ranks {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.rank-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: var(--color-bg);
  border-radius: 8px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  transition: var(--transition);
}

.rank-row.winner {
  background: rgba(108, 99, 255, 0.1);
  border: 1px solid rgba(108, 99, 255, 0.3);
}

.rank-num {
  min-width: 30px;
  color: var(--color-text-light);
  font-weight: 600;
}

.rank-digit {
  font-size: 18px;
  font-weight: 800;
  min-width: 28px;
  text-align: center;
}

.rank-prob {
  min-width: 70px;
  font-weight: 700;
  color: var(--color-text);
}

.rank-bar-container {
  flex: 1;
  height: 12px;
  background: var(--color-border);
  border-radius: 6px;
  overflow: hidden;
}

.rank-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--color-primary), var(--color-accent));
  border-radius: 6px;
  transition: width 0.5s ease;
}

.rank-row.winner .rank-bar {
  background: linear-gradient(90deg, var(--color-success), var(--color-accent));
}
</style>
