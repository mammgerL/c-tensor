<script setup>
import { computed, ref } from 'vue'

const currentStep = ref(0)

const trainingSteps = [
  {
    id: 'forward',
    label: '前向传播',
    title: 'Forward Pass: 构建预测',
    desc: '输入像素先映射到隐藏层，再通过激活函数与输出层得到 10 个类别分数。',
    math: 'out = log_softmax(ReLU(xW1 + b1)W2 + b2)',
    codeLines: [3, 9],
    details: [
      { title: '矩阵乘法', text: '784 维输入与权重矩阵相乘，把像素空间压缩成更适合分类的特征表示。' },
      { title: '激活函数', text: 'ReLU 截断负值，给网络引入非线性，否则两层线性层仍然等价于一层。' },
    ],
    visualType: 'forward-flow',
  },
  {
    id: 'loss',
    label: '计算损失',
    title: 'Loss Calculation: 衡量误差',
    desc: 'NLL Loss 会直接惩罚“正确类别概率不够高”的预测结果。',
    math: 'L = -sum(y_i * log(p_i))',
    codeLines: [12, 12],
    details: [
      { title: '监督信号', text: '标签以 one-hot 形式给出，损失函数会把预测分布和真实类别对齐。' },
      { title: '优化目标', text: '训练的本质就是不断降低这一个标量 Loss。' },
    ],
    visualType: 'loss-dist',
  },
  {
    id: 'backward',
    label: '反向传播',
    title: 'Backward Pass: 计算梯度',
    desc: '链式法则会把输出误差逐层回传，得到每个参数应该如何调整的梯度。',
    math: 'grad(W) = dL / dW',
    codeLines: [15, 15],
    details: [
      { title: '责任分摊', text: '梯度表示“这个参数变大一点，Loss 会朝哪个方向变化”。' },
      { title: '自动求导', text: 'tensor.h 在前向过程中记录依赖关系，backward() 会沿图回传梯度。' },
    ],
    visualType: 'backward-flow',
  },
  {
    id: 'update',
    label: '参数更新',
    title: 'SGD Update: 修正权重',
    desc: '拿到梯度后，SGD 会沿着负梯度方向移动参数，让下次预测更接近真实标签。',
    math: 'W = W - lr * grad(W)',
    codeLines: [18, 26],
    details: [
      { title: '学习率', text: '学习率控制每一步走多远。过大容易震荡，过小则收敛太慢。' },
      { title: '清理状态', text: '更新后要清零梯度并释放前向图里的临时张量，避免梯度累积和内存泄漏。' },
    ],
    visualType: 'weight-update',
  },
]

const currentStepData = computed(() => trainingSteps[currentStep.value])

const trainCode = `// train.c 核心循环
for (int step = 0; step < TRAIN_STEPS; step++) {
    Tensor *x = tensor_from_arr(batch_x);
    Tensor *h1 = matmul(x, w1);
    Tensor *h1b = add_bias(h1, b1);
    Tensor *r1 = relu(h1b);
    Tensor *h2 = matmul(r1, w2);
    Tensor *h2b = add_bias(h2, b2);
    Tensor *out = logsoftmax(h2b);

    Tensor *loss = nll_loss(out, batch_y);

    backward(loss);

    for (int i = 0; i < w1->data->size; i++) {
        w1->data->values[i] -= lr * w1->grad->values[i];
    }
    for (int i = 0; i < b1->data->size; i++) {
        b1->data->values[i] -= lr * b1->grad->values[i];
    }
    for (int i = 0; i < w2->data->size; i++) {
        w2->data->values[i] -= lr * w2->grad->values[i];
    }
    for (int i = 0; i < b2->data->size; i++) {
        b2->data->values[i] -= lr * b2->grad->values[i];
    }

    zero_grad(w1); zero_grad(b1);
    zero_grad(w2); zero_grad(b2);
    free_tensor(loss); free_tensor(out); // 其余中间张量同理
}`
</script>

<template>
  <div class="training-view">
    <header class="page-header">
      <h1>训练过程演示</h1>
      <p class="page-desc">把一次训练迭代拆成 4 个关键阶段，直接对照仓库里的 C 实现。</p>
    </header>

    <div class="main-layout">
      <section class="viz-panel">
        <div class="step-nav">
          <button
            v-for="(step, index) in trainingSteps"
            :key="step.id"
            :class="['nav-btn', { active: currentStep === index }]"
            @click="currentStep = index"
          >
            <span class="step-idx">{{ index + 1 }}</span>
            <span>{{ step.label }}</span>
          </button>
        </div>

        <div class="viz-card">
          <div class="viz-header">
            <div>
              <p class="step-tag">Step {{ currentStep + 1 }}</p>
              <h2>{{ currentStepData.title }}</h2>
            </div>
            <div class="math-badge">{{ currentStepData.math }}</div>
          </div>

          <p class="viz-desc">{{ currentStepData.desc }}</p>

          <div class="visual-area">
            <div v-if="currentStepData.visualType === 'forward-flow'" class="flow-container">
              <div class="flow-node input">Input</div>
              <div class="flow-arrow forward">→</div>
              <div class="flow-node hidden">Hidden</div>
              <div class="flow-arrow forward">→</div>
              <div class="flow-node output">Output</div>
            </div>

            <div v-else-if="currentStepData.visualType === 'loss-dist'" class="loss-container">
              <div class="dist-box">
                <div class="dist-bar" :style="{ height: '72%' }">预测</div>
                <div class="dist-bar target" :style="{ height: '100%' }">标签</div>
              </div>
              <div class="loss-arrow">比较分布差异，得到 Loss</div>
            </div>

            <div v-else-if="currentStepData.visualType === 'backward-flow'" class="flow-container">
              <div class="flow-node output active-back">Output</div>
              <div class="flow-arrow backward">←</div>
              <div class="flow-node hidden active-back">Hidden</div>
              <div class="flow-arrow backward">←</div>
              <div class="flow-node input active-back">Input</div>
            </div>

            <div v-else class="update-container">
              <div class="weight-grid">
                <div v-for="cell in 16" :key="cell" class="weight-cell pulse"></div>
              </div>
              <div class="update-text">W = W - lr * grad</div>
            </div>
          </div>

          <div class="details-grid">
            <article v-for="detail in currentStepData.details" :key="detail.title" class="detail-item">
              <h3>{{ detail.title }}</h3>
              <p>{{ detail.text }}</p>
            </article>
          </div>
        </div>
      </section>

      <aside class="code-panel">
        <div class="code-header">
          <h3>对应 C 实现</h3>
          <span class="code-hint">高亮行和当前步骤对应</span>
        </div>

        <div class="code-container">
          <pre class="code-content"><div
            v-for="(line, index) in trainCode.split('\n')"
            :key="index"
            :class="['code-line', { highlight: index + 1 >= currentStepData.codeLines[0] && index + 1 <= currentStepData.codeLines[1] }]"
          ><span class="line-num">{{ index + 1 }}</span><span class="line-text">{{ line }}</span></div></pre>
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.training-view {
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 24px;
}

.page-header {
  text-align: center;
  margin-bottom: 40px;
}

.page-header h1 {
  margin: 0 0 12px;
  font-size: 34px;
  font-weight: 800;
}

.page-desc {
  margin: 0;
  font-size: 17px;
  color: var(--color-text-light);
}

.main-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 430px;
  gap: 28px;
}

.viz-panel,
.code-panel {
  min-width: 0;
}

.step-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 24px;
}

.nav-btn {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 12px 18px;
  border: 1px solid var(--color-border);
  border-radius: 14px;
  background: var(--color-card);
  color: var(--color-text-light);
  cursor: pointer;
  font-size: 14px;
  font-weight: 700;
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.nav-btn:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.nav-btn.active {
  color: white;
  background: var(--color-primary);
  border-color: var(--color-primary);
}

.step-idx {
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.18);
  font-size: 12px;
}

.viz-card {
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-height: 620px;
  padding: 32px;
  border-radius: 24px;
  background: var(--color-card);
  box-shadow: var(--shadow-md);
}

.viz-header {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: flex-start;
}

.step-tag {
  margin: 0 0 8px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--color-primary);
}

.viz-header h2 {
  margin: 0;
  font-size: 26px;
}

.math-badge {
  flex-shrink: 0;
  padding: 10px 14px;
  border: 1px solid var(--color-border);
  border-radius: 12px;
  background: var(--color-bg);
  font-family: "SF Mono", Monaco, monospace;
  font-size: 13px;
}

.viz-desc {
  margin: 0;
  font-size: 17px;
  line-height: 1.7;
  color: var(--color-text-light);
}

.visual-area {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 280px;
  padding: 36px;
  border-radius: 20px;
  background: var(--color-bg);
}

.flow-container {
  display: flex;
  align-items: center;
  gap: 22px;
}

.flow-node {
  min-width: 96px;
  padding: 18px 22px;
  border: 2px solid var(--color-border);
  border-radius: 14px;
  background: var(--color-card);
  text-align: center;
  font-weight: 800;
}

.flow-node.input {
  border-color: #4f9cf9;
}

.flow-node.hidden {
  border-color: #f59e0b;
}

.flow-node.output {
  border-color: #22c55e;
}

.flow-arrow {
  font-size: 34px;
  font-weight: 900;
}

.flow-arrow.forward {
  color: var(--color-primary);
  animation: slideRight 1.5s infinite;
}

.flow-arrow.backward {
  color: #ef4444;
  animation: slideLeft 1.5s infinite;
}

.active-back {
  border-color: #ef4444;
  color: #ef4444;
}

.loss-container,
.update-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
}

.dist-box {
  display: flex;
  align-items: flex-end;
  gap: 20px;
  height: 160px;
}

.dist-bar {
  width: 72px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding-bottom: 10px;
  border-radius: 12px 12px 0 0;
  background: var(--color-primary);
  color: white;
  font-size: 12px;
  font-weight: 800;
}

.dist-bar.target {
  background: #22c55e;
}

.loss-arrow,
.update-text {
  font-family: "SF Mono", Monaco, monospace;
  font-size: 14px;
  font-weight: 800;
  color: var(--color-text);
}

.weight-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}

.weight-cell {
  width: 44px;
  height: 44px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-card);
}

.pulse {
  animation: pulseBg 1.8s infinite;
}

.details-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.detail-item {
  padding: 18px;
  border-left: 4px solid var(--color-primary);
  border-radius: 14px;
  background: var(--color-bg);
}

.detail-item h3 {
  margin: 0 0 8px;
  font-size: 16px;
}

.detail-item p {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--color-text-light);
}

.code-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.code-header h3 {
  margin: 0;
  font-size: 18px;
}

.code-hint {
  font-size: 12px;
  font-weight: 700;
  color: var(--color-primary);
}

.code-container {
  overflow: auto;
  padding: 20px;
  border-radius: 20px;
  background: #111827;
  box-shadow: var(--shadow-md);
}

.code-content {
  margin: 0;
  font-family: "SF Mono", Monaco, monospace;
  font-size: 13px;
  line-height: 1.65;
  color: #d1d5db;
}

.code-line {
  display: flex;
  gap: 14px;
  padding: 0 8px;
  border-radius: 6px;
}

.code-line.highlight {
  background: rgba(79, 156, 249, 0.18);
  outline: 1px solid rgba(79, 156, 249, 0.28);
}

.line-num {
  width: 28px;
  flex-shrink: 0;
  text-align: right;
  color: #6b7280;
  user-select: none;
}

.line-text {
  white-space: pre;
}

@keyframes slideRight {
  0% { transform: translateX(-10px); opacity: 0.3; }
  50% { opacity: 1; }
  100% { transform: translateX(10px); opacity: 0.3; }
}

@keyframes slideLeft {
  0% { transform: translateX(10px); opacity: 0.3; }
  50% { opacity: 1; }
  100% { transform: translateX(-10px); opacity: 0.3; }
}

@keyframes pulseBg {
  0%, 100% { background: var(--color-card); }
  50% { background: rgba(79, 156, 249, 0.22); }
}

@media (max-width: 1100px) {
  .main-layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .training-view {
    padding: 28px 16px;
  }

  .viz-card {
    padding: 24px;
  }

  .viz-header {
    flex-direction: column;
  }

  .details-grid {
    grid-template-columns: 1fr;
  }

  .flow-container {
    gap: 14px;
  }

  .flow-node {
    min-width: 74px;
    padding: 14px 10px;
    font-size: 13px;
  }
}
</style>
