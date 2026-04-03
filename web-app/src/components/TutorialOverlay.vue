<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({
  show: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['close'])
const currentStep = ref(0)

const steps = [
  {
    title: '欢迎来到 AI 小学堂！🎉',
    content: '在这里，你会学到人工智能的神奇知识，还能亲手画出数字让 AI 来猜！准备好了吗？',
    emoji: '🚀',
  },
  {
    title: '什么是神经网络？🧠',
    content: '就像你的大脑有很多神经元一样，神经网络也有很多"小神经元"！它们一起工作来认识世界。',
    emoji: '💡',
  },
  {
    title: '今天我们要学什么？📋',
    content: '1️⃣ 学习神经网络基础知识\n2️⃣ 看看 AI 怎么认数字\n3️⃣ 自己画数字让 AI 猜\n4️⃣ 探索更多有趣的 AI 知识',
    emoji: '📚',
  },
  {
    title: '准备好了吗？出发！🌟',
    content: '点击导航栏开始你的 AI 冒险之旅吧！每个页面都有好玩的东西等着你哦~',
    emoji: '🎮',
  },
]

function next() {
  if (currentStep.value < steps.length - 1) {
    currentStep.value++
  } else {
    emit('close')
  }
}

function prev() {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

function close() {
  emit('close')
}
</script>

<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="show" class="tutorial-overlay" @click.self="close">
        <div class="tutorial-card animate-slide-in">
          <button class="close-btn" @click="close">✕</button>

          <div class="step-indicator">
            <span
              v-for="(_, i) in steps"
              :key="i"
              :class="['dot', { active: i === currentStep, completed: i < currentStep }]"
            ></span>
          </div>

          <div class="step-content">
            <span class="step-emoji">{{ steps[currentStep].emoji }}</span>
            <h2>{{ steps[currentStep].title }}</h2>
            <p class="step-text">{{ steps[currentStep].content }}</p>
          </div>

          <div class="step-nav">
            <button
              v-if="currentStep > 0"
              class="btn-prev"
              @click="prev"
            >
              ← 上一步
            </button>
            <button class="btn-next" @click="next">
              {{ currentStep === steps.length - 1 ? '开始探索 🚀' : '下一步 →' }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.tutorial-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
}

.tutorial-card {
  background: var(--color-card);
  border-radius: var(--radius-lg);
  padding: 40px;
  max-width: 500px;
  width: 100%;
  position: relative;
  box-shadow: var(--shadow-lg);
}

.close-btn {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--color-bg);
  color: var(--color-text-light);
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  background: var(--color-danger);
  color: white;
}

.step-indicator {
  display: flex;
  gap: 8px;
  justify-content: center;
  margin-bottom: 24px;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--color-border);
  transition: var(--transition);
}

.dot.active {
  background: var(--color-primary);
  transform: scale(1.3);
}

.dot.completed {
  background: var(--color-success);
}

.step-content {
  text-align: center;
}

.step-emoji {
  font-size: 64px;
  display: block;
  margin-bottom: 16px;
  animation: float 3s ease-in-out infinite;
}

.step-content h2 {
  font-size: 24px;
  font-weight: 800;
  color: var(--color-text);
  margin-bottom: 16px;
}

.step-text {
  font-size: 16px;
  line-height: 1.6;
  color: var(--color-text-light);
  white-space: pre-line;
}

.step-nav {
  display: flex;
  justify-content: space-between;
  margin-top: 32px;
  gap: 12px;
}

.btn-prev,
.btn-next {
  padding: 12px 24px;
  border-radius: var(--radius-sm);
  font-weight: 700;
  font-size: 15px;
}

.btn-prev {
  background: var(--color-bg);
  color: var(--color-text-light);
}

.btn-next {
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  color: white;
  flex: 1;
}

.btn-next:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}
</style>