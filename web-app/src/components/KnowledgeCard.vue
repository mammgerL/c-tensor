<script setup>
import { ref } from 'vue'

const props = defineProps({
  title: {
    type: String,
    required: true,
  },
  icon: {
    type: String,
    default: '💡',
  },
  difficulty: {
    type: String,
    default: 'easy',
  },
})

const isExpanded = ref(false)

const difficultyLabels = {
  easy: '🌱 简单',
  medium: '🌿 中等',
  hard: '🌳 进阶',
}

const difficultyColors = {
  easy: '#4CAF50',
  medium: '#FFC107',
  hard: '#FF5252',
}
</script>

<template>
  <div class="knowledge-card" :class="{ expanded: isExpanded }" @click="isExpanded = !isExpanded">
    <div class="card-header">
      <span class="card-icon">{{ icon }}</span>
      <div class="card-title-area">
        <h3 class="card-title">{{ title }}</h3>
        <span
          class="difficulty-badge"
          :style="{ background: difficultyColors[difficulty] + '22', color: difficultyColors[difficulty] }"
        >
          {{ difficultyLabels[difficulty] }}
        </span>
      </div>
      <span class="expand-icon">{{ isExpanded ? '▲' : '▼' }}</span>
    </div>
    <Transition name="expand">
      <div v-if="isExpanded" class="card-body">
        <slot></slot>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.knowledge-card {
  background: var(--color-card);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  cursor: pointer;
  transition: var(--transition);
  overflow: hidden;
}

.knowledge-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px 24px;
}

.card-icon {
  font-size: 36px;
  animation: float 3s ease-in-out infinite;
}

.card-title-area {
  flex: 1;
}

.card-title {
  font-size: 18px;
  font-weight: 700;
  color: var(--color-text);
  margin-bottom: 4px;
}

.difficulty-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
}

.expand-icon {
  font-size: 14px;
  color: var(--color-text-light);
  transition: var(--transition);
}

.card-body {
  padding: 0 24px 20px;
  font-size: 15px;
  line-height: 1.7;
  color: var(--color-text-light);
}

.expand-enter-active,
.expand-leave-active {
  transition: all 0.3s ease;
}

.expand-enter-from,
.expand-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>