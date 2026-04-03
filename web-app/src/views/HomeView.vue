<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const features = [
  {
    icon: '📐',
    title: '矩阵计算演示',
    desc: '手写一个数字，逐步查看 784→256→10 的矩阵乘法、ReLU、LogSoftmax 计算过程',
    path: '/playground',
    color: '#6C63FF',
  },
  {
    icon: '⚙️',
    title: 'C 代码原理',
    desc: '了解 tensor.h 的核心数据结构、自动微分、Kaiming 初始化等实现细节',
    path: '/learn',
    color: '#FF6B9D',
  },
  {
    icon: '🔍',
    title: '数据探索',
    desc: '浏览 MNIST 测试集，分析哪些数字容易被误识别，理解模型边界情况',
    path: '/explore',
    color: '#00D2FF',
  },
]

const codeSteps = [
  { step: 1, label: '加载数据', code: 'fscanf(file, "%f", &pixel)' },
  { step: 2, label: '矩阵乘法', code: 'h1 = matmul(x, w1)  // [1,784] × [784,256]' },
  { step: 3, label: '加偏置', code: 'h1b = add_bias(h1, b1)  // + [256]' },
  { step: 4, label: 'ReLU 激活', code: 'r1 = relu(h1b)  // max(0, x)' },
  { step: 5, label: '第二层', code: 'h2 = matmul(r1, w2)  // [1,256] × [256,10]' },
  { step: 6, label: 'LogSoftmax', code: 'out = logsoftmax(h2b)  // 概率分布' },
  { step: 7, label: 'Argmax', code: 'predicted = argmax(out)  // 最大概率的索引' },
]

const currentCodeStep = ref(0)

function navigate(path) {
  router.push(path)
}
</script>

<template>
  <div class="home-view">
    <section class="hero-section">
      <div class="hero-content">
        <h1 class="hero-title">
          <span class="title-text">C-Tensor</span>
        </h1>
        <p class="hero-subtitle">纯 C 实现的神经网络，从零理解 MNIST 手写数字识别</p>
        <p class="hero-desc">
          这个项目用纯 C 语言实现了一个包含自动微分的张量库，<br>
          训练了一个 784→256→10 的多层感知机来识别手写数字。<br>
          在这里你可以可视化地看到每一步矩阵计算是怎么进行的。
        </p>
        <div class="hero-actions">
          <button class="btn-primary btn-lg" @click="navigate('/playground')">
            📐 开始计算演示
          </button>
          <button class="btn-secondary btn-lg" @click="navigate('/learn')">
            ⚙️ 查看 C 代码原理
          </button>
        </div>
      </div>
    </section>

    <section class="pipeline-section">
      <h2 class="section-title">前向传播流水线</h2>
      <div class="pipeline-steps">
        <div
          v-for="(item, i) in codeSteps"
          :key="item.step"
          :class="['pipeline-step', { active: i === currentCodeStep }]"
          @click="currentCodeStep = i"
        >
          <span class="step-num">{{ item.step }}</span>
          <span class="step-label">{{ item.label }}</span>
          <code class="step-code">{{ item.code }}</code>
        </div>
      </div>
    </section>

    <section class="features-section">
      <h2 class="section-title">选择你要探索的方向</h2>
      <div class="features-grid">
        <div
          v-for="feature in features"
          :key="feature.path"
          class="feature-card"
          @click="navigate(feature.path)"
        >
          <div class="feature-icon">{{ feature.icon }}</div>
          <h3 class="feature-title">{{ feature.title }}</h3>
          <p class="feature-desc">{{ feature.desc }}</p>
          <div class="feature-arrow" :style="{ color: feature.color }">→</div>
        </div>
      </div>
    </section>

    <section class="architecture-section">
      <h2 class="section-title">网络架构</h2>
      <div class="arch-diagram">
        <div class="arch-layer arch-input">
          <span class="arch-label">输入层</span>
          <span class="arch-shape">[1, 784]</span>
          <span class="arch-desc">28×28 像素展平</span>
        </div>
        <div class="arch-arrow">
          <span class="arch-op">matmul</span>
          <span class="arch-w">W1 [784, 256]</span>
        </div>
        <div class="arch-layer arch-hidden">
          <span class="arch-label">隐藏层</span>
          <span class="arch-shape">[1, 256]</span>
          <span class="arch-desc">ReLU 激活</span>
        </div>
        <div class="arch-arrow">
          <span class="arch-op">matmul</span>
          <span class="arch-w">W2 [256, 10]</span>
        </div>
        <div class="arch-layer arch-output">
          <span class="arch-label">输出层</span>
          <span class="arch-shape">[1, 10]</span>
          <span class="arch-desc">LogSoftmax</span>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.home-view {
  max-width: 1100px;
  margin: 0 auto;
  padding: 40px 24px;
}

.hero-section {
  margin-bottom: 60px;
}

.hero-title {
  margin-bottom: 12px;
}

.title-text {
  font-size: 48px;
  font-weight: 800;
  background: linear-gradient(135deg, #6C63FF, #00D2FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 20px;
  font-weight: 600;
  color: var(--color-text-light);
  margin-bottom: 16px;
}

.hero-desc {
  font-size: 15px;
  line-height: 1.7;
  color: var(--color-text-light);
  margin-bottom: 28px;
}

.hero-actions {
  display: flex;
  gap: 16px;
}

.btn-lg {
  padding: 14px 28px;
  font-size: 16px;
}

.section-title {
  font-size: 24px;
  font-weight: 800;
  text-align: center;
  margin-bottom: 28px;
}

.pipeline-section {
  margin-bottom: 60px;
}

.pipeline-steps {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.pipeline-step {
  display: grid;
  grid-template-columns: 36px 100px 1fr;
  align-items: center;
  gap: 16px;
  padding: 12px 20px;
  background: var(--color-card);
  border-radius: 12px;
  cursor: pointer;
  transition: var(--transition);
  border-left: 3px solid transparent;
}

.pipeline-step:hover {
  box-shadow: var(--shadow-sm);
}

.pipeline-step.active {
  border-left-color: var(--color-primary);
  background: rgba(108, 99, 255, 0.05);
}

.step-num {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--color-bg);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  color: var(--color-text-light);
}

.pipeline-step.active .step-num {
  background: var(--color-primary);
  color: white;
}

.step-label {
  font-weight: 600;
  font-size: 14px;
}

.step-code {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 4px 10px;
  border-radius: 6px;
}

.features-section {
  margin-bottom: 60px;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.feature-card {
  background: var(--color-card);
  border-radius: 16px;
  padding: 28px 24px;
  text-align: center;
  cursor: pointer;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
}

.feature-card:hover {
  transform: translateY(-6px);
  box-shadow: var(--shadow-lg);
}

.feature-icon {
  font-size: 40px;
  margin-bottom: 14px;
  display: block;
}

.feature-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 8px;
}

.feature-desc {
  font-size: 13px;
  color: var(--color-text-light);
  line-height: 1.6;
  margin-bottom: 14px;
}

.feature-arrow {
  font-size: 22px;
  font-weight: 700;
  transition: var(--transition);
}

.feature-card:hover .feature-arrow {
  transform: translateX(6px);
}

.architecture-section {
  margin-bottom: 60px;
}

.arch-diagram {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.arch-layer {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 20px 24px;
  border-radius: 14px;
  min-width: 120px;
}

.arch-input {
  background: rgba(108, 99, 255, 0.1);
  border: 2px solid rgba(108, 99, 255, 0.3);
}

.arch-hidden {
  background: rgba(255, 107, 157, 0.1);
  border: 2px solid rgba(255, 107, 157, 0.3);
}

.arch-output {
  background: rgba(0, 210, 255, 0.1);
  border: 2px solid rgba(0, 210, 255, 0.3);
}

.arch-label {
  font-size: 14px;
  font-weight: 700;
}

.arch-shape {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--color-text-light);
}

.arch-desc {
  font-size: 12px;
  color: var(--color-text-light);
}

.arch-arrow {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  color: var(--color-text-light);
  font-size: 13px;
}

.arch-op {
  font-weight: 700;
  font-size: 14px;
}

.arch-w {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px;
  color: var(--color-text-light);
}

@media (max-width: 768px) {
  .features-grid {
    grid-template-columns: 1fr;
  }

  .arch-diagram {
    flex-direction: column;
  }

  .title-text {
    font-size: 36px;
  }
}
</style>
