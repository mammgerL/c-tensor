<script setup>
import { ref, computed, onMounted } from 'vue'

const props = defineProps({
  output: {
    type: Array,
    required: true,
  },
})

const probabilities = computed(() => {
  const probs = props.output.map((v, i) => ({ digit: i, prob: Math.exp(v) }))
  const sum = probs.reduce((a, b) => a + b.prob, 0)
  probs.forEach(p => p.prob = p.prob / sum)
  probs.sort((a, b) => b.prob - a.prob)
  return probs
})

const digitEmojis = ['0️⃣', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣']

const barColors = [
  '#6C63FF', '#FF6B9D', '#00D2FF', '#4CAF50', '#FFC107',
  '#FF5252', '#9C27B0', '#00BCD4', '#FF9800', '#8BC34A'
]
</script>

<template>
  <div class="probability-chart">
    <h3 class="chart-title">🎯 AI 认为每个数字的可能性</h3>
    <div class="prob-list">
      <div
        v-for="(item, index) in probabilities"
        :key="item.digit"
        class="prob-item"
        :class="{ 'is-top': index === 0 }"
      >
        <div class="prob-digit">
          <span class="digit-emoji">{{ digitEmojis[item.digit] }}</span>
          <span class="digit-number">{{ item.digit }}</span>
        </div>
        <div class="prob-bar-container">
          <div class="prob-bar-bg">
            <div
              class="prob-bar-fill"
              :style="{
                width: (item.prob * 100) + '%',
                background: `linear-gradient(90deg, ${barColors[item.digit]}, ${barColors[item.digit]}dd)`
              }"
            ></div>
          </div>
        </div>
        <div class="prob-value">{{ (item.prob * 100).toFixed(1) }}%</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.probability-chart {
  width: 100%;
}

.chart-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 16px;
  color: var(--color-text);
}

.prob-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.prob-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  border-radius: 12px;
  transition: var(--transition);
}

.prob-item.is-top {
  background: var(--color-bg);
}

.prob-digit {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 40px;
}

.digit-emoji {
  font-size: 20px;
}

.digit-number {
  font-size: 12px;
  font-weight: 700;
  color: var(--color-text-light);
}

.prob-bar-container {
  flex: 1;
}

.prob-bar-bg {
  height: 24px;
  background: var(--color-bg);
  border-radius: 12px;
  overflow: hidden;
}

.prob-bar-fill {
  height: 100%;
  border-radius: 12px;
  transition: width 0.5s ease;
  min-width: 2px;
}

.prob-value {
  width: 55px;
  text-align: right;
  font-weight: 700;
  font-size: 14px;
  color: var(--color-text-light);
}
</style>