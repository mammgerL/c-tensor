<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  pixels: {
    type: Array,
    required: true,
  },
  trueLabel: {
    type: Number,
    required: true,
  },
  predicted: {
    type: Number,
    required: true,
  },
  correct: {
    type: Boolean,
    required: true,
  },
  confidence: {
    type: Number,
    required: true,
  },
})

const canvasRef = ref(null)

watch(() => props.pixels, () => {
  drawImage()
}, { immediate: true })

function drawImage() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const size = 28
  const scale = 8

  ctx.fillStyle = 'white'
  ctx.fillRect(0, 0, 224, 224)

  const imageData = ctx.createImageData(size, size)
  for (let i = 0; i < props.pixels.length; i++) {
    const normalized = (props.pixels[i] + 1) / 2
    const pixel = Math.max(0, Math.min(255, Math.round(normalized * 255)))
    imageData.data[i * 4] = pixel
    imageData.data[i * 4 + 1] = pixel
    imageData.data[i * 4 + 2] = pixel
    imageData.data[i * 4 + 3] = 255
  }

  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = size
  tempCanvas.height = size
  tempCanvas.getContext('2d').putImageData(imageData, 0, 0)

  ctx.imageSmoothingEnabled = false
  ctx.drawImage(tempCanvas, 0, 0, size * scale, size * scale)
}
</script>

<template>
  <div class="digit-display">
    <canvas ref="canvasRef" width="224" height="224" class="digit-canvas"></canvas>
    <div class="digit-info">
      <div class="digit-labels">
        <div class="label-item">
          <span class="label-icon">📝</span>
          <span class="label-text">真实数字: <strong>{{ trueLabel }}</strong></span>
        </div>
        <div class="label-item">
          <span class="label-icon">🤖</span>
          <span class="label-text">AI 猜的是: <strong>{{ predicted }}</strong></span>
        </div>
      </div>
      <div :class="['result-badge', correct ? 'correct' : 'incorrect']">
        {{ correct ? '✅ 猜对啦！' : '❌ 猜错了...' }}
      </div>
      <div class="confidence-bar">
        <div class="confidence-label">置信度</div>
        <div class="confidence-track">
          <div
            class="confidence-fill"
            :style="{ width: (confidence * 100) + '%' }"
          ></div>
        </div>
        <div class="confidence-value">{{ (confidence * 100).toFixed(1) }}%</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.digit-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.digit-canvas {
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-md);
  image-rendering: pixelated;
}

.digit-info {
  width: 100%;
  text-align: center;
}

.digit-labels {
  display: flex;
  gap: 24px;
  justify-content: center;
  margin-bottom: 12px;
}

.label-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
}

.label-icon {
  font-size: 20px;
}

.label-text strong {
  color: var(--color-primary);
  font-size: 20px;
}

.result-badge {
  display: inline-block;
  padding: 8px 20px;
  border-radius: 20px;
  font-weight: 700;
  font-size: 16px;
  margin-bottom: 16px;
}

.result-badge.correct {
  background: rgba(76, 175, 80, 0.15);
  color: var(--color-success);
}

.result-badge.incorrect {
  background: rgba(255, 82, 82, 0.15);
  color: var(--color-danger);
}

.confidence-bar {
  margin-top: 8px;
}

.confidence-label {
  font-size: 14px;
  color: var(--color-text-light);
  margin-bottom: 6px;
}

.confidence-track {
  height: 12px;
  background: var(--color-bg);
  border-radius: 6px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--color-primary), var(--color-accent));
  border-radius: 6px;
  transition: width 0.5s ease;
}

.confidence-value {
  font-size: 14px;
  font-weight: 700;
  color: var(--color-primary);
  margin-top: 4px;
}
</style>