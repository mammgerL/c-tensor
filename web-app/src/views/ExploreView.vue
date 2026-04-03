<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import DigitDisplay from '../components/DigitDisplay.vue'
import ProbabilityChart from '../components/ProbabilityChart.vue'
import ActivationHeatmap from '../components/ActivationHeatmap.vue'
import NetworkVisual from '../components/NetworkVisual.vue'

const currentIndex = ref(0)
const currentData = ref(null)
const isLoading = ref(false)
const filter = ref('all')
const stats = ref(null)
const sampleGrid = ref([])
const showGrid = ref(true)

const networkLayers = [
  { name: 'input', label: '👀 输入层', size: 16, x: 80 },
  { name: 'hidden', label: '🧠 隐藏层', size: 16, x: 300 },
  { name: 'output', label: '🎯 输出层', size: 10, x: 520 },
]

onMounted(async () => {
  await loadStats()
  await loadSampleGrid()
  await loadSample(0)
})

async function loadStats() {
  try {
    const res = await fetch('/api/eval')
    if (res.ok) stats.value = await res.json()
  } catch (e) {
    console.error('Failed to load stats:', e)
  }
}

async function loadSampleGrid() {
  try {
    const promises = []
    for (let i = 0; i < 50; i++) {
      const idx = Math.floor(Math.random() * 10000)
      promises.push(fetch(`/api/predict?index=${idx}`).then(r => r.json()).catch(() => null))
    }
    const results = await Promise.all(promises)
    sampleGrid.value = results.filter(Boolean).map((d, i) => ({ ...d, gridIndex: i }))
  } catch (e) {
    console.error('Failed to load sample grid:', e)
  }
}

async function loadSample(index) {
  isLoading.value = true
  try {
    const res = await fetch(`/api/predict?index=${index}`)
    if (!res.ok) throw new Error('Failed')
    currentData.value = await res.json()
    currentIndex.value = index
  } catch (e) {
    console.error('Failed to load sample:', e)
  } finally {
    isLoading.value = false
  }
}

function navigate(dir) {
  const next = currentIndex.value + dir
  if (next >= 0 && next < 10000) {
    loadSample(next)
  }
}

function randomSample() {
  loadSample(Math.floor(Math.random() * 10000))
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
      <p class="page-desc">浏览 10000 个手写数字，看看 AI 是怎么认出来的！</p>
    </header>

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
            max="9999"
            class="index-input"
          />
          <span class="index-range">/ 10000</span>
        </div>
        <button class="nav-btn" @click="navigate(1)" :disabled="currentIndex >= 9999">下一个 →</button>
        <button class="random-btn" @click="randomSample">🎲 随机</button>
      </div>

      <div v-if="isLoading" class="loading-state">
        <span class="loading-spinner">⏳</span>
        <p>加载中...</p>
      </div>

      <div v-else-if="currentData" class="sample-detail">
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