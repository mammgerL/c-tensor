<script setup>
import { ref, computed, onMounted, watch, nextTick } from 'vue'

const props = defineProps({
  hidden: {
    type: Array,
    default: () => [],
  },
})

const canvasRef = ref(null)
const hoveredIndex = ref(-1)

const stats = computed(() => {
  if (!props.hidden.length) return null
  const active = props.hidden.filter(v => v > 0).length
  const dead = props.hidden.length - active
  const max = Math.max(...props.hidden)
  const mean = props.hidden.reduce((a, b) => a + b, 0) / props.hidden.length
  return { active, dead, total: props.hidden.length, max, mean }
})

const hoveredInfo = computed(() => {
  if (hoveredIndex.value < 0 || !props.hidden.length) return null
  const val = props.hidden[hoveredIndex.value]
  return {
    index: hoveredIndex.value,
    value: val,
    isActive: val > 0,
  }
})

// Color gradient: 0 → dark, max → bright cyan/blue
function valueToColor(val, maxVal) {
  if (val <= 0) return { r: 20, g: 20, b: 30 } // dead neuron: very dark
  const t = val / maxVal
  // Dark blue → Cyan → White
  const r = Math.round(20 + t * 180)
  const g = Math.round(20 + t * 220)
  const b = Math.round(60 + t * 195)
  return { r, g, b }
}

function drawHeatmap() {
  const canvas = canvasRef.value
  if (!canvas || !props.hidden.length) return
  const ctx = canvas.getContext('2d')

  const cols = 16
  const rows = Math.ceil(props.hidden.length / cols)
  const cellW = canvas.width / cols
  const cellH = canvas.height / rows
  const maxVal = Math.max(...props.hidden, 0.001)

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  props.hidden.forEach((val, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    const { r, g, b } = valueToColor(val, maxVal)

    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
    const x = col * cellW
    const y = row * cellH
    // Slight gap between cells
    ctx.beginPath()
    ctx.roundRect(x + 1, y + 1, cellW - 2, cellH - 2, 2)
    ctx.fill()
  })
}

function handleCanvasHover(e) {
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const scaleX = canvas.width / rect.width
  const scaleY = canvas.height / rect.height
  const x = (e.clientX - rect.left) * scaleX
  const y = (e.clientY - rect.top) * scaleY

  const cols = 16
  const cellW = canvas.width / cols
  const cellH = canvas.height / Math.ceil(props.hidden.length / cols)
  const col = Math.floor(x / cellW)
  const row = Math.floor(y / cellH)
  const idx = row * cols + col

  if (idx >= 0 && idx < props.hidden.length) {
    hoveredIndex.value = idx
  } else {
    hoveredIndex.value = -1
  }
}

function handleCanvasLeave() {
  hoveredIndex.value = -1
}

onMounted(() => {
  nextTick(drawHeatmap)
})

watch(() => props.hidden, () => {
  nextTick(drawHeatmap)
})
</script>

<template>
  <div class="activation-heatmap">
    <h3 class="heatmap-title">隐藏层神经元激活图</h3>
    <p class="heatmap-desc">256 个神经元 (16x16)，亮度 = 激活强度，暗色 = 被 ReLU 关闭的神经元</p>

    <div class="heatmap-body">
      <canvas
        ref="canvasRef"
        width="320"
        height="320"
        class="heatmap-canvas"
        @mousemove="handleCanvasHover"
        @mouseleave="handleCanvasLeave"
      ></canvas>

      <div class="heatmap-sidebar">
        <!-- Color scale -->
        <div class="color-scale">
          <div class="scale-bar"></div>
          <div class="scale-labels">
            <span>max</span>
            <span>0</span>
          </div>
        </div>

        <!-- Stats -->
        <div class="heatmap-stats" v-if="stats">
          <div class="stat-row active">
            <span class="stat-dot active-dot"></span>
            <span>激活: <strong>{{ stats.active }}</strong></span>
          </div>
          <div class="stat-row dead">
            <span class="stat-dot dead-dot"></span>
            <span>沉默: <strong>{{ stats.dead }}</strong></span>
          </div>
          <div class="stat-row">
            <span class="stat-label-text">最大值:</span>
            <strong>{{ stats.max.toFixed(2) }}</strong>
          </div>
          <div class="stat-row">
            <span class="stat-label-text">平均值:</span>
            <strong>{{ stats.mean.toFixed(2) }}</strong>
          </div>
        </div>

        <!-- Hover tooltip -->
        <div class="neuron-tooltip" v-if="hoveredInfo">
          <span class="tooltip-label">神经元 #{{ hoveredInfo.index }}</span>
          <span class="tooltip-value" :class="{ active: hoveredInfo.isActive }">
            {{ hoveredInfo.isActive ? hoveredInfo.value.toFixed(4) : '0 (沉默)' }}
          </span>
        </div>
        <div class="neuron-tooltip placeholder" v-else>
          <span class="tooltip-label">鼠标悬停查看神经元</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.activation-heatmap {
  width: 100%;
}

.heatmap-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 6px;
}

.heatmap-desc {
  font-size: 13px;
  color: var(--color-text-light);
  margin-bottom: 16px;
}

.heatmap-body {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

.heatmap-canvas {
  width: 280px;
  height: 280px;
  border-radius: 10px;
  border: 1px solid var(--color-border);
  background: rgb(20, 20, 30);
  cursor: crosshair;
  flex-shrink: 0;
}

.heatmap-sidebar {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 130px;
}

/* Color scale */
.color-scale {
  display: flex;
  gap: 8px;
  align-items: stretch;
}

.scale-bar {
  width: 16px;
  height: 100px;
  border-radius: 8px;
  background: linear-gradient(
    to bottom,
    rgb(200, 240, 255),
    rgb(100, 180, 200),
    rgb(40, 80, 120),
    rgb(20, 20, 30)
  );
  border: 1px solid var(--color-border);
}

.scale-labels {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  font-size: 11px;
  color: var(--color-text-light);
  font-weight: 600;
}

/* Stats */
.heatmap-stats {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stat-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--color-text-light);
}

.stat-dot {
  width: 10px;
  height: 10px;
  border-radius: 3px;
  flex-shrink: 0;
}

.active-dot {
  background: rgb(100, 200, 220);
}

.dead-dot {
  background: rgb(40, 40, 50);
  border: 1px solid var(--color-border);
}

.stat-label-text {
  font-size: 12px;
}

/* Neuron tooltip */
.neuron-tooltip {
  background: var(--color-bg);
  padding: 10px 12px;
  border-radius: 10px;
  font-size: 13px;
}

.neuron-tooltip.placeholder {
  opacity: 0.5;
}

.tooltip-label {
  display: block;
  font-weight: 700;
  margin-bottom: 4px;
  color: var(--color-text);
}

.tooltip-value {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-weight: 600;
  color: var(--color-text-light);
}

.tooltip-value.active {
  color: rgb(100, 220, 240);
}

@media (max-width: 600px) {
  .heatmap-body {
    flex-direction: column;
    align-items: center;
  }

  .heatmap-canvas {
    width: 100%;
    height: auto;
    aspect-ratio: 1;
  }

  .heatmap-sidebar {
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: center;
  }
}
</style>
