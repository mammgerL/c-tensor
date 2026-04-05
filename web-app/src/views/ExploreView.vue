<script setup>
import { ref, onMounted } from 'vue'
import DigitDisplay from '../components/DigitDisplay.vue'
import ProbabilityChart from '../components/ProbabilityChart.vue'
import ActivationHeatmap from '../components/ActivationHeatmap.vue'
import NetworkVisual from '../components/NetworkVisual.vue'
import { MnistInference, loadTestSamples, normalizePixels } from '../inference.js'
import weightsUrl from '../assets/weights.bin?url'

// In dev mode, load the full 10k test set; in production builds (e.g. GitHub
// Pages) load a smaller 1k subset to keep the deployed bundle compact.
// Vite treeshakes the unused dynamic import at build time via the static env check.
const inference = new MnistInference()

const currentIndex = ref(0)
const currentData = ref(null)
const isLoading = ref(true)
const loadMessage = ref('加载权重和测试集…')
const stats = ref(null)
const sampleGrid = ref([])
const sampleCount = ref(0)

// In-memory test data + slim prediction cache
let samples = null  // { count, pixelsU8: Uint8Array, labels: Uint8Array }
const predictions = []  // [{ predicted, correct }, ...], same length as samples.count

onMounted(async () => {
  try {
    loadMessage.value = '加载权重…'
    await inference.loadWeights(weightsUrl)

    loadMessage.value = '加载测试样例…'
    const samplesUrl = import.meta.env.DEV
      ? (await import('../assets/test_samples_10000.bin?url')).default
      : (await import('../assets/test_samples_1000.bin?url')).default
    samples = await loadTestSamples(samplesUrl)
    sampleCount.value = samples.count

    loadMessage.value = `推理 ${samples.count.toLocaleString()} 个样例…`
    await scoreAll()

    loadSampleGrid()
    loadSample(0)
  } catch (e) {
    console.error('Failed to initialize explore view:', e)
    loadMessage.value = `加载失败：${e.message}`
  } finally {
    isLoading.value = false
  }
})

async function scoreAll() {
  // Batched loop with yields every few hundred samples so the UI thread
  // can update the "loading" label and stay responsive on large sets.
  const N = samples.count
  let correct = 0
  const CHUNK = 500
  for (let start = 0; start < N; start += CHUNK) {
    const end = Math.min(start + CHUNK, N)
    for (let i = start; i < end; i++) {
      const x = normalizePixels(samples.pixelsU8, i * 784, 784)
      const { predicted } = inference.predictFast(x)
      const isCorrect = predicted === samples.labels[i]
      predictions[i] = { predicted, correct: isCorrect }
      if (isCorrect) correct++
    }
    loadMessage.value = `推理中 ${end.toLocaleString()} / ${N.toLocaleString()}…`
    await new Promise(r => setTimeout(r, 0))  // yield to paint loop
  }
  stats.value = {
    total: N,
    correct,
    incorrect: N - correct,
    accuracy: (correct / N) * 100,
  }
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
    const p = predictions[idx]
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
        <span class="page-desc-env">
          ({{ sampleCount >= 10000 ? '本地开发模式：全量 10,000 个样例' : '生产模式：1,000 个样例' }})
        </span>
      </p>
    </header>

    <div v-if="isLoading" class="init-loading">
      <span class="loading-spinner">⏳</span>
      <p>{{ loadMessage }}</p>
    </div>

    <template v-else>

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

.page-desc-env {
  display: inline-block;
  margin-left: 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--color-primary);
  padding: 2px 10px;
  border-radius: 999px;
  background: rgba(108, 99, 255, 0.1);
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
.random-btn {
  padding: 10px 20px;
  border-radius: var(--radius-sm);
  font-weight: 600;
  font-size: 14px;
  background: var(--color-bg);
  color: var(--color-text);
}

.nav-btn:hover,
.random-btn:hover {
  background: var(--color-border);
}

.nav-btn:disabled {
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