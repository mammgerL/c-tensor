<script setup>
import { ref, onMounted } from 'vue'
import DigitDisplay from '../components/DigitDisplay.vue'
import ProbabilityChart from '../components/ProbabilityChart.vue'
import ActivationHeatmap from '../components/ActivationHeatmap.vue'
import NetworkVisual from '../components/NetworkVisual.vue'
import { MnistInference, loadTestSamples, normalizePixels } from '../inference.js'

const inference = new MnistInference()
const DATASET = {
  key: 'mnist-mini',
  name: 'MNIST 测试集',
  description: '线上线下统一加载 1000 条 MNIST 测试样本，配套使用轻量 MNIST 模型，避免 GitHub Pages 首屏加载过重。',
  weightsUrl: './weights.bin',
  loadUrl: async () => (await import('../assets/test_samples_1000.bin?url')).default,
}

const currentIndex = ref(0)
const currentData = ref(null)
const isLoading = ref(true)
const loadMessage = ref('加载权重和测试集…')
const stats = ref(null)
const sampleGrid = ref([])
const sampleCount = ref(0)
const scoringProgress = ref(null)  // { done, total } while background scoring runs

// In-memory test data + slim prediction cache
let samples = null  // { count, pixelsU8: Uint8Array, labels: Uint8Array }
let predictions = []  // [{ predicted, correct }, ...], same length as samples.count
let datasetLoadVersion = 0

onMounted(async () => {
  try {
    await loadCurrentDataset()
  } catch (e) {
    console.error('Failed to initialize explore view:', e)
    loadMessage.value = `加载失败：${e.message}`
    isLoading.value = false
  }
})

function resetDatasetState() {
  samples = null
  predictions = []
  currentIndex.value = 0
  currentData.value = null
  stats.value = null
  sampleGrid.value = []
  sampleCount.value = 0
  scoringProgress.value = null
}

async function loadCurrentDataset() {
  const version = ++datasetLoadVersion
  isLoading.value = true
  resetDatasetState()
  loadMessage.value = `加载 ${DATASET.name}（1000 条样本）与对应模型…`

  await inference.loadWeights(DATASET.weightsUrl)
  if (version !== datasetLoadVersion) return

  const samplesUrl = await DATASET.loadUrl()
  const loadedSamples = await loadTestSamples(samplesUrl)
  if (version !== datasetLoadVersion) return

  samples = loadedSamples
  sampleCount.value = samples.count
  isLoading.value = false

  loadSample(0)
  loadSampleGrid()
  scoreAllInBackground(version)
}

function scoreSample(i) {
  // Memoized slim forward pass; returns cached entry if already computed.
  if (predictions[i]) return predictions[i]
  const x = normalizePixels(samples.pixelsU8, i * 784, 784)
  const { predicted } = inference.predictFast(x)
  const entry = { predicted, correct: predicted === samples.labels[i] }
  predictions[i] = entry
  return entry
}

async function scoreAllInBackground(version) {
  const N = sampleCount.value
  scoringProgress.value = { done: 0, total: N }
  let correct = 0
  const CHUNK = 250
  for (let start = 0; start < N; start += CHUNK) {
    if (version !== datasetLoadVersion) return
    const end = Math.min(start + CHUNK, N)
    for (let i = start; i < end; i++) {
      if (scoreSample(i).correct) correct++
    }
    scoringProgress.value = { done: end, total: N }
    await new Promise(r => setTimeout(r, 0))  // yield to paint loop
  }
  if (version !== datasetLoadVersion) return
  // Recompute correct count from cache (scoreSample may have counted some
  // during grid/navigation, but we haven't been summing those — easier to
  // re-sum once at the end).
  correct = 0
  for (let i = 0; i < N; i++) if (predictions[i].correct) correct++
  stats.value = {
    total: N,
    correct,
    incorrect: N - correct,
    accuracy: (correct / N) * 100,
  }
  scoringProgress.value = null
}

function predictDetailed(index) {
  const x = normalizePixels(samples.pixelsU8, index * 784, 784)
  const result = inference.predict(x)
  return {
    index,
    true_label: samples.labels[index],
    predicted: result.predicted,
    correct: result.predicted === samples.labels[index],
    confidence: result.confidence,
    pixels: result.pixels,
    hidden: result.hidden,
    pre_relu: result.pre_relu,
    output: result.output,
    pre_softmax: result.pre_softmax,
  }
}

function loadSample(index) {
  if (index < 0 || index >= sampleCount.value) return
  currentData.value = predictDetailed(index)
  currentIndex.value = index
}

function loadSampleGrid() {
  const picks = new Set()
  const count = Math.min(50, sampleCount.value)
  while (picks.size < count) {
    picks.add(Math.floor(Math.random() * sampleCount.value))
  }
  sampleGrid.value = Array.from(picks).map((idx, gridIndex) => {
    const p = scoreSample(idx)
    return {
      gridIndex,
      index: idx,
      true_label: samples.labels[idx],
      predicted: p.predicted,
      correct: p.correct,
    }
  })
}

function navigate(dir) {
  const next = currentIndex.value + dir
  if (next >= 0 && next < sampleCount.value) {
    loadSample(next)
  }
}

function randomSample() {
  loadSample(Math.floor(Math.random() * sampleCount.value))
}

function nextIncorrect() {
  // Search forward from the current index, wrapping around, for the next
  // sample the model got wrong.
  const N = sampleCount.value
  for (let step = 1; step <= N; step++) {
    const idx = (currentIndex.value + step) % N
    if (!predictions[idx].correct) {
      loadSample(idx)
      return
    }
  }
}

function randomIncorrect() {
  // Collect all incorrect indices and pick one at random.
  const wrongs = []
  for (let i = 0; i < sampleCount.value; i++) {
    if (!predictions[i].correct) wrongs.push(i)
  }
  if (wrongs.length > 0) {
    loadSample(wrongs[Math.floor(Math.random() * wrongs.length)])
  }
}

function selectFromGrid(item) {
  if (item && item.index !== undefined) {
    loadSample(item.index)
  }
}
</script>

<template>
  <div class="explore-view">
    <header class="page-header">
      <h1>🔍 探索数据集</h1>
      <p class="page-desc">
        浏览
        <strong>{{ sampleCount.toLocaleString() }}</strong>
        个手写数字，看看 AI 是怎么认出来的！
      </p>
      <div class="dataset-summary">
        <span class="dataset-chip">固定资源集</span>
        <p class="dataset-description">{{ DATASET.description }}</p>
      </div>
    </header>

    <div v-if="isLoading" class="init-loading">
      <span class="loading-spinner">⏳</span>
      <p>{{ loadMessage }}</p>
    </div>

    <template v-else>

    <div v-if="scoringProgress" class="scoring-progress">
      <span class="scoring-spinner">⚡</span>
      <span class="scoring-label">
        后台计算准确率：{{ scoringProgress.done.toLocaleString() }} / {{ scoringProgress.total.toLocaleString() }}
      </span>
      <div class="scoring-bar">
        <div class="scoring-bar-fill" :style="{ width: (scoringProgress.done / scoringProgress.total * 100) + '%' }"></div>
      </div>
    </div>

    <div v-if="stats" class="stats-banner">
      <div class="stat-item">
        <span class="stat-icon">📊</span>
        <div class="stat-info">
          <span class="stat-label">总样本</span>
          <span class="stat-value">{{ stats.total }}</span>
        </div>
      </div>
      <div class="stat-item">
        <span class="stat-icon">✅</span>
        <div class="stat-info">
          <span class="stat-label">正确</span>
          <span class="stat-value" style="color: var(--color-success)">{{ stats.correct }}</span>
        </div>
      </div>
      <div class="stat-item">
        <span class="stat-icon">❌</span>
        <div class="stat-info">
          <span class="stat-label">错误</span>
          <span class="stat-value" style="color: var(--color-danger)">{{ stats.incorrect }}</span>
        </div>
      </div>
      <div class="stat-item">
        <span class="stat-icon">🎯</span>
        <div class="stat-info">
          <span class="stat-label">准确率</span>
          <span class="stat-value">{{ stats.accuracy.toFixed(2) }}%</span>
        </div>
      </div>
    </div>

    <div class="sample-grid-section">
      <div class="section-header">
        <h2>🎲 随机样本</h2>
        <button class="btn-primary" @click="loadSampleGrid">🔄 刷新</button>
      </div>
      <div class="sample-grid">
        <div
          v-for="item in sampleGrid"
          :key="item.gridIndex"
          :class="['sample-thumb', { correct: item.correct, incorrect: !item.correct }]"
          @click="selectFromGrid(item)"
        >
          <div class="thumb-label">{{ item.true_label }}</div>
          <div class="thumb-predict">AI: {{ item.predicted }}</div>
        </div>
      </div>
    </div>

    <div class="explore-main">
      <div class="sample-nav">
        <button class="nav-btn" @click="navigate(-1)" :disabled="currentIndex <= 0">← 上一个</button>
        <div class="index-display">
          <span class="index-label">当前样本</span>
          <input
            type="number"
            v-model.number="currentIndex"
            @change="loadSample(currentIndex)"
            min="0"
            :max="sampleCount - 1"
            class="index-input"
          />
          <span class="index-range">/ {{ sampleCount.toLocaleString() }}</span>
        </div>
        <button class="nav-btn" @click="navigate(1)" :disabled="currentIndex >= sampleCount - 1">下一个 →</button>
        <button class="random-btn" @click="randomSample">🎲 随机</button>
        <button
          class="wrong-btn"
          @click="nextIncorrect"
          :disabled="!stats || stats.incorrect === 0"
          title="跳到下一个分类错误的样本"
        >
          ❌ 下一个错误
        </button>
        <button
          class="wrong-btn"
          @click="randomIncorrect"
          :disabled="!stats || stats.incorrect === 0"
          title="随机跳到一个分类错误的样本"
        >
          🎯 随机错例
        </button>
      </div>

      <div v-if="currentData" class="sample-detail">
        <div class="detail-grid">
          <div class="detail-left">
            <DigitDisplay
              :pixels="currentData.pixels"
              :true-label="currentData.true_label"
              :predicted="currentData.predicted"
              :correct="currentData.correct"
              :confidence="currentData.confidence"
            />
          </div>
          <div class="detail-right">
            <ProbabilityChart :output="currentData.output" />
          </div>
        </div>

        <div class="detail-bottom">
          <div class="network-section">
            <h3 class="section-title">神经网络连接与激活</h3>
            <p class="section-desc">线条亮度 = 连接强度，节点颜色 = 激活程度，绿色 = 预测结果</p>
            <NetworkVisual
              :step="7"
              :hiddenActivations="currentData.hidden"
              :outputActivations="currentData.output"
              :preSoftmax="currentData.pre_softmax || []"
              :preRelu="currentData.pre_relu || []"
              :predicted="currentData.predicted"
            />
          </div>
          <div class="heatmap-section">
            <ActivationHeatmap :hidden="currentData.hidden" />
          </div>
        </div>
      </div>
    </div>

    </template>
  </div>
</template>

<style scoped>
.explore-view {
  max-width: 1100px;
  margin: 0 auto;
  padding: 40px 24px;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.page-header h1 {
  font-size: 36px;
  font-weight: 800;
  margin-bottom: 8px;
}

.page-desc {
  font-size: 18px;
  color: var(--color-text-light);
}

.dataset-description {
  margin: 0;
  max-width: 560px;
  font-size: 14px;
  color: var(--color-text-light);
}

.dataset-summary {
  margin: 18px auto 0;
  max-width: 560px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.dataset-chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(108, 99, 255, 0.1);
  color: var(--color-primary);
  font-size: 12px;
  font-weight: 800;
}

.init-loading {
  text-align: center;
  padding: 80px 20px;
}

.init-loading p {
  margin-top: 16px;
  font-size: 15px;
  color: var(--color-text-light);
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.scoring-progress {
  display: flex;
  align-items: center;
  gap: 10px;
  background: rgba(108, 99, 255, 0.06);
  border: 1px solid rgba(108, 99, 255, 0.15);
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 16px;
  font-size: 13px;
}

.scoring-spinner {
  font-size: 16px;
  animation: pulse 1.2s ease-in-out infinite;
}

.scoring-label {
  color: var(--color-text-light);
  font-family: 'SF Mono', 'Fira Code', monospace;
  flex-shrink: 0;
}

.scoring-bar {
  flex: 1;
  height: 6px;
  background: rgba(108, 99, 255, 0.12);
  border-radius: 3px;
  overflow: hidden;
}

.scoring-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #6C63FF, #00D2FF);
  transition: width 0.15s ease;
}

.stats-banner {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 40px;
}

.stat-item {
  background: var(--color-card);
  border-radius: var(--radius-md);
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: var(--shadow-sm);
}

.stat-icon {
  font-size: 32px;
}

.stat-label {
  display: block;
  font-size: 13px;
  color: var(--color-text-light);
  margin-bottom: 4px;
}

.stat-value {
  font-size: 24px;
  font-weight: 800;
}

.sample-grid-section {
  margin-bottom: 40px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section-header h2 {
  font-size: 22px;
  font-weight: 700;
}

.sample-grid {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 8px;
}

.sample-thumb {
  aspect-ratio: 1;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: var(--transition);
  border: 2px solid transparent;
  font-size: 12px;
  font-weight: 600;
}

.sample-thumb.correct {
  background: rgba(76, 175, 80, 0.1);
  border-color: rgba(76, 175, 80, 0.3);
}

.sample-thumb.incorrect {
  background: rgba(255, 82, 82, 0.1);
  border-color: rgba(255, 82, 82, 0.3);
}

.sample-thumb:hover {
  transform: scale(1.1);
  box-shadow: var(--shadow-md);
}

.thumb-label {
  font-size: 16px;
  font-weight: 800;
}

.thumb-predict {
  font-size: 10px;
  color: var(--color-text-light);
}

.explore-main {
  background: var(--color-card);
  border-radius: var(--radius-md);
  padding: 24px;
  box-shadow: var(--shadow-sm);
}

.sample-nav {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
  flex-wrap: wrap;
}

.nav-btn,
.random-btn,
.wrong-btn {
  padding: 10px 20px;
  border-radius: var(--radius-sm);
  font-weight: 600;
  font-size: 14px;
  background: var(--color-bg);
  color: var(--color-text);
}

.wrong-btn {
  background: rgba(255, 82, 82, 0.08);
  color: var(--color-danger, #FF5252);
  border: 1px solid rgba(255, 82, 82, 0.25);
}

.nav-btn:hover,
.random-btn:hover {
  background: var(--color-border);
}

.wrong-btn:hover:not(:disabled) {
  background: rgba(255, 82, 82, 0.16);
}

.nav-btn:disabled,
.wrong-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.index-display {
  display: flex;
  align-items: center;
  gap: 8px;
}

.index-label {
  font-size: 14px;
  color: var(--color-text-light);
}

.index-input {
  width: 70px;
  padding: 8px 12px;
  border: 2px solid var(--color-border);
  border-radius: 8px;
  font-size: 16px;
  font-weight: 700;
  text-align: center;
  font-family: inherit;
}

.index-input:focus {
  outline: none;
  border-color: var(--color-primary);
}

.index-range {
  font-size: 14px;
  color: var(--color-text-light);
}

.loading-state {
  text-align: center;
  padding: 60px;
}

.loading-spinner {
  font-size: 48px;
  display: block;
  animation: pulse 1s ease-in-out infinite;
}

.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
  margin-bottom: 32px;
}

.detail-bottom {
  margin-top: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.network-section,
.heatmap-section {
  background: var(--color-bg);
  border-radius: var(--radius-md);
  padding: 20px;
}

.section-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 4px;
}

.section-desc {
  font-size: 13px;
  color: var(--color-text-light);
  margin-bottom: 16px;
}

@media (max-width: 768px) {
  .stats-banner {
    grid-template-columns: repeat(2, 1fr);
  }

  .sample-grid {
    grid-template-columns: repeat(5, 1fr);
  }

  .detail-grid {
    grid-template-columns: 1fr;
  }
}
</style>
