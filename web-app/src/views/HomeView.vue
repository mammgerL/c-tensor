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

const intuitions = [
  {
    num: '01',
    title: '一个"神经元"就是一个数字',
    body: '输入层的 784 个神经元，是 28×28 每个像素的亮度 (0 ~ 1)。输出层的 10 个神经元，分别代表"这是 0 的信心、是 1 的信心…是 9 的信心"。隐藏层的 256 个神经元呢？—— 那正是网络要"学出来"的东西。',
    visual: 'neurons',
  },
  {
    num: '02',
    title: '每个神经元在"寻找"一种模式',
    body: '一个隐藏层神经元接收全部 784 个输入，每个输入配一个权重 w。正权重说"我在乎这里亮"，负权重说"我在乎这里暗"，接近 0 说"我不关心"。把这 784 个权重排回 28×28，就是这个神经元在"找"的那张图。',
    visual: 'formula',
    formulaText: 'h = ReLU( Σ wᵢ · xᵢ + b )',
  },
  {
    num: '03',
    title: '矩阵乘法 = 256 个神经元同时思考',
    body: 'matmul(x, W1) 不是什么数学把戏 —— 它就是 256 个隐藏层神经元同时对同一张图做加权和。W1 的每一列就是一个神经元的权重向量。深度学习离不开矩阵，是因为矩阵就是"并行的思考"。',
    visual: 'formula',
    formulaText: 'x [1,784]  ×  W1 [784,256]  =  h [1,256]',
  },
  {
    num: '04',
    title: '整个网络只是 203,530 个旋钮',
    body: '784×256 (W1) + 256 (b1) + 256×10 (W2) + 10 (b2) = 203,530 个参数。训练就是在这 203,530 维空间里，找到一组让模型答对的数字。"梯度下降"就是在问：每个旋钮该往哪拧？拧多少？',
    visual: 'params',
  },
]

const principles = [
  {
    num: '01',
    title: '归一化像素',
    shape: '[1, 784]',
    why: '像素值从 0 到 255 差距太大，直接喂进网络会让前几步的加权和数值爆炸、梯度也不稳定。',
    how: '除以 127.5 再减 1，把每个像素压到 [−1, 1]。黑色背景 (0) 变成 −1，白色笔画 (255) 变成 +1，灰色抗锯齿像素落在中间。',
    code: 'x[i] = pixel / 127.5 - 1.0',
    example: `一个亮度 230 的灰色像素：
x = 230 / 127.5 − 1 = 0.804

黑色背景像素 (0) 会变成 −1，纯白笔画 (255) 会变成 +1。`,
  },
  {
    num: '02',
    title: '第一次矩阵乘法 · 256 个模式探测器同时工作',
    shape: '[1, 784] → [1, 256]',
    why: '784 个原始像素太低级了，我们需要从里面抽出"更有意义"的特征 —— 边缘、弧线、笔画方向…',
    how: '256 个神经元各自对 784 个像素做一次加权和，总共 784 × 256 = 200,704 次乘加。W1 的每一列，就是一个神经元想找的那张图。',
    code: 'h1 = matmul(x, W1)   // x @ W1',
    example: `拿第 0 号隐藏神经元来算：
h1[0] = x[0]·W1[0,0] + x[1]·W1[1,0] + ... + x[783]·W1[783,0]
      = (−1.0)·(−0.04) + (−1.0)·(0.02) + ... + 0.804·0.12 + ...
      = 2.05

784 项相加就是这一个数。256 个神经元一起做，就得到一个 256 维向量。`,
  },
  {
    num: '03',
    title: '加偏置 · 每个神经元的激活门槛',
    shape: '[1, 256]',
    why: '有些模式只要稍微像就该触发，有些要非常像才算数 —— 神经元之间的"挑剔程度"不该是一样的。',
    how: '给 256 个神经元各加一个可学习的 b。b 越负，这个神经元越挑剔；越正，越容易被点亮。',
    code: 'h1 = h1 + b1',
    example: `继续上面那个神经元：
h1[0] = 2.05 + b1[0]
      = 2.05 + 0.08
      = 2.13

如果 b1[0] = −5，这个神经元哪怕看到 2.05 的匹配度也会被压到负数，下一步 ReLU 就会把它关掉。`,
  },
  {
    num: '04',
    title: 'ReLU · 没看到就闭嘴',
    shape: '[1, 256]',
    why: '如果每一层都是线性的，堆再多层也只是一次线性变换。网络必须有"非线性决策"的能力。',
    how: '负值归零。就这么一个简单操作，让每个神经元能自由地"开/关"，256 个神经元组合出指数级的决策空间。',
    code: 'h1 = max(0, h1)',
    example: `假设前 5 个神经元加完偏置后是：
h1 = [ 2.13, −0.47,  0.88, −1.20,  3.02, ... ]
   → [ 2.13,    0  ,  0.88,    0  ,  3.02, ... ]

第 1、3 号神经元"关闭"（归零），剩下的保持不变。256 个里通常有一半左右会被 ReLU 关掉。`,
  },
  {
    num: '05',
    title: '第二次矩阵乘法 · 把特征翻译成数字',
    shape: '[1, 256] → [1, 10]',
    why: '现在我们有 256 个"抽象特征"，但用户要的是 0~9 中的一个答案。需要把特征再组合一次。',
    how: '10 个输出神经元，每个对 256 个特征做加权和（再加 b2）。W2 的第 c 列告诉我们"数字 c 喜欢哪些特征"。',
    code: 'h2 = matmul(h1, W2) + b2',
    example: `对"是不是 7"的那个输出（索引 7）：
h2[7] = h1[0]·W2[0,7] + h1[1]·W2[1,7] + ... + h1[255]·W2[255,7] + b2[7]
      = 2.13·0.41 + 0·(−0.22) + 0.88·0.67 + ... + 0.08
      = 5.12

10 个类别同时算完，得到 10 个原始分数（logits）：
h2 = [−1.02, −0.38, 2.15, 0.47, −2.33, 1.20, −1.85,  5.12, −0.76, 2.04]
       0      1     2     3     4      5     6       7      8     9
"7"的分数最高，其它类别被压得很低。`,
  },
  {
    num: '06',
    title: 'LogSoftmax · 把原始分数变成概率',
    shape: '[1, 10]',
    why: '10 个原始分数不能直接当信心用。直接 exp/sum 又会数值溢出（exp(100) 就爆了）。',
    how: '先减去最大值再做 log-sum-exp：数值稳定，并且和训练时的 NLL loss 天然配合 (log-prob 相加即可)。',
    code: 'out[c] = h2[c] − max − log(Σ exp(h2 − max))',
    example: `h2 = [−1.02, −0.38, 2.15, 0.47, −2.33, 1.20, −1.85, 5.12, −0.76, 2.04]
max = 5.12

1. 每个值减去 max：
   centered = [−6.14, −5.50, −2.97, ..., 0.00, ..., −3.08]
2. 算 log(Σ exp(centered))：
   Σ exp = 1.00 + 0.0513 + 0.00957 + ... ≈ 1.135
   log(1.135) = 0.127
3. 每个值都减这个 log-sum-exp：
   out[7] = 5.12 − 5.12 − 0.127 = −0.127

out[c] 是对数概率，exp(out[c]) 是真正的概率，所有类别概率加起来 = 1。`,
  },
  {
    num: '07',
    title: 'Argmax · 模型的最终答案',
    shape: '[1, 10] → 1',
    why: '用户要的是"这是几"，不是一个概率分布。',
    how: '选最大的那个索引。对应的 exp(out[c]) 就是模型的"信心"。',
    code: 'predicted = argmax(out)',
    example: `out = [−6.27, −5.63, −3.10, −4.78, −7.58,
       −4.05, −7.10, −0.127, −6.01, −3.21]
         0      1     2     3     4
         5     6     7      8     9

最大值在索引 7 → predicted = 7
信心 = exp(−0.127) ≈ 0.881 = 88.1%

意思是：模型 88.1% 确定这是一个 7，剩下的 11.9% 分散在其它 9 个数字上（主要是 2 和 9，因为它们的 out 值相对较高）。`,
  },
]

const weights = [
  {
    name: 'W1',
    shape: '[784, 256]',
    role: '第一层模式探测器：每一列是一个神经元在 784 个像素上的权重',
    min: '−0.167',
    max: '+0.160',
    count: '200,704',
  },
  {
    name: 'b1',
    shape: '[256]',
    role: '256 个隐藏神经元各自的激活门槛',
    min: '−0.008',
    max: '+0.007',
    count: '256',
  },
  {
    name: 'W2',
    shape: '[256, 10]',
    role: '特征 → 数字的映射：每一列告诉模型"数字 c 喜欢哪些隐藏特征"',
    min: '−0.363',
    max: '+0.399',
    count: '2,560',
  },
  {
    name: 'b2',
    shape: '[10]',
    role: '10 个数字类别各自的偏好（prior）',
    min: '−0.008',
    max: '+0.010',
    count: '10',
  },
]

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
          用纯 C 实现一个带自动微分的张量库，训练 784 → 256 → 10 的多层感知机识别手写数字。<br>
          这一页把神经网络的每一步拆开来讲 —— 先理解直觉，再看具体发生了什么。
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

    <section class="intuition-section">
      <h2 class="section-title">神经网络到底在做什么？</h2>
      <p class="section-subtitle">
        先抛开公式，理解直觉。下面 4 条借鉴了 3Blue1Brown《Neural Networks》系列的讲法。
      </p>
      <div class="intuition-grid">
        <div v-for="item in intuitions" :key="item.num" class="intuition-card">
          <div class="intuition-num">{{ item.num }}</div>
          <h3 class="intuition-title">{{ item.title }}</h3>
          <p class="intuition-body">{{ item.body }}</p>
          <div v-if="item.visual === 'neurons'" class="intuition-visual neurons-row">
            <span v-for="(op, i) in [0.08, 0.22, 0.55, 0.88, 0.7, 0.38, 0.15, 0.05]" :key="i" class="neuron-dot" :style="{ opacity: op }"></span>
          </div>
          <div v-else-if="item.visual === 'formula'" class="intuition-visual formula-box">{{ item.formulaText }}</div>
          <div v-else-if="item.visual === 'params'" class="intuition-visual params-box">
            <span class="param-number">203,530</span>
            <span class="param-label">个可学习参数</span>
          </div>
        </div>
      </div>
      <div class="intuition-honest">
        <span class="honest-badge">诚实地说</span>
        <p>
          理想情况下，第一层识别笔画边缘，第二层识别圈和弧线，最后组合成数字 —— 但实际训练出来的权重图更像噪声，
          神经元学到的是我们说不清楚的特征。而它就是有效。
          这就是深度学习让人又爱又困惑的地方。
        </p>
      </div>
    </section>

    <section class="principles-section">
      <h2 class="section-title">跟着一张 "7" 走一遍网络</h2>
      <p class="section-subtitle">
        从像素到答案，前向传播的每一步都在解决一个具体问题。左边是"为什么"，右边是"怎么做"。
      </p>
      <div class="principles-list">
        <article v-for="item in principles" :key="item.num" class="principle-card">
          <header class="principle-head">
            <span class="principle-num">{{ item.num }}</span>
            <h3 class="principle-title">{{ item.title }}</h3>
            <span class="principle-shape">{{ item.shape }}</span>
          </header>
          <div class="principle-body">
            <div class="principle-col">
              <span class="col-label">问题</span>
              <p>{{ item.why }}</p>
            </div>
            <div class="principle-col">
              <span class="col-label">做法</span>
              <p>{{ item.how }}</p>
            </div>
          </div>
          <code class="principle-code">{{ item.code }}</code>
          <details v-if="item.example" class="principle-example">
            <summary>举个具体例子 ▾</summary>
            <pre>{{ item.example }}</pre>
          </details>
        </article>
      </div>
    </section>

    <section class="weights-section">
      <h2 class="section-title">打开黑盒：网络里到底存了什么？</h2>
      <p class="section-subtitle">
        "训练好的网络"其实就是下面这 4 个张量里的 203,530 个具体浮点数。下面是本项目真实训练后的数据。
      </p>
      <div class="weights-grid">
        <div v-for="w in weights" :key="w.name" class="weight-card">
          <div class="weight-head">
            <span class="weight-name">{{ w.name }}</span>
            <span class="weight-shape">{{ w.shape }}</span>
          </div>
          <p class="weight-role">{{ w.role }}</p>
          <div class="weight-stats">
            <div class="weight-stat">
              <span class="stat-label">数值范围</span>
              <span class="stat-value">{{ w.min }} ~ {{ w.max }}</span>
            </div>
            <div class="weight-stat">
              <span class="stat-label">参数个数</span>
              <span class="stat-value">{{ w.count }}</span>
            </div>
          </div>
        </div>
      </div>
      <div class="weights-notes">
        <h3 class="notes-title">从这组数字里能看出什么？</h3>
        <ul>
          <li><strong>b1、b2 都非常接近 0</strong>（最大也就 ±0.01）—— 训练出来的神经元"挑剔程度"其实都差不多，偏置只做微调。</li>
          <li><strong>W1 的值域 (±0.16) 比 W2 (±0.40) 小</strong> —— Kaiming 初始化按 <code>√(2/fan_in)</code> 缩放，784 个输入的扇入让 W1 一开始就被压得比较小，训练过程也保持了这个量级。</li>
          <li><strong>W1 每一列排回 28×28 后，并不像边缘或笔画检测器</strong> —— 更像是有一定结构的噪声。这就是 3Blue1Brown 系列里说的"愿望 vs 现实"。</li>
          <li>想亲眼看看这些数字怎么工作？<a href="#" @click.prevent="navigate('/playground')">去 Playground</a> 画一个数字，每一个神经元被激活了多少都是实时计算出来的。</li>
        </ul>
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

.section-subtitle {
  text-align: center;
  font-size: 14px;
  color: var(--color-text-light);
  margin: -20px auto 28px;
  max-width: 560px;
  line-height: 1.6;
}

.intuition-section {
  margin-bottom: 60px;
}

.intuition-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.intuition-card {
  position: relative;
  background: var(--color-card);
  border-radius: 16px;
  padding: 28px 26px 24px;
  box-shadow: var(--shadow-sm);
  transition: var(--transition);
  display: flex;
  flex-direction: column;
}

.intuition-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.intuition-num {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  font-weight: 700;
  color: var(--color-primary);
  letter-spacing: 1px;
  margin-bottom: 10px;
}

.intuition-title {
  font-size: 17px;
  font-weight: 700;
  margin-bottom: 12px;
  line-height: 1.4;
}

.intuition-body {
  font-size: 14px;
  line-height: 1.75;
  color: var(--color-text-light);
  margin-bottom: 16px;
  flex: 1;
}

.intuition-visual {
  margin-top: auto;
  padding-top: 12px;
  border-top: 1px dashed var(--color-border, rgba(0,0,0,0.08));
}

.neurons-row {
  display: flex;
  gap: 6px;
  align-items: center;
  justify-content: flex-start;
  padding-top: 14px;
}

.neuron-dot {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--color-primary);
}

.formula-box {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  color: var(--color-text);
  background: var(--color-bg);
  padding: 10px 12px;
  border-radius: 8px;
  margin-top: 12px;
  text-align: center;
}

.params-box {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding-top: 14px;
}

.param-number {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 26px;
  font-weight: 800;
  background: linear-gradient(135deg, #6C63FF, #00D2FF);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
}

.param-label {
  font-size: 12px;
  color: var(--color-text-light);
}

.intuition-honest {
  display: flex;
  gap: 14px;
  align-items: flex-start;
  background: rgba(255, 107, 157, 0.06);
  border-left: 3px solid #FF6B9D;
  border-radius: 10px;
  padding: 16px 18px;
}

.intuition-honest p {
  font-size: 14px;
  line-height: 1.75;
  color: var(--color-text-light);
  margin: 0;
}

.honest-badge {
  flex-shrink: 0;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: #FF6B9D;
  background: rgba(255, 107, 157, 0.12);
  padding: 4px 10px;
  border-radius: 6px;
  margin-top: 2px;
}

.principles-section {
  margin-bottom: 60px;
}

.principles-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.principle-card {
  background: var(--color-card);
  border-radius: 14px;
  padding: 22px 24px 18px;
  box-shadow: var(--shadow-sm);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
}

.principle-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, #6C63FF, #00D2FF);
  opacity: 0;
  transition: var(--transition);
}

.principle-card:hover {
  transform: translateX(3px);
  box-shadow: var(--shadow-lg);
}

.principle-card:hover::before {
  opacity: 1;
}

.principle-head {
  display: flex;
  align-items: baseline;
  gap: 14px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}

.principle-num {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  font-weight: 700;
  color: var(--color-primary);
  letter-spacing: 1px;
  flex-shrink: 0;
}

.principle-title {
  font-size: 16px;
  font-weight: 700;
  margin: 0;
  flex: 1;
  line-height: 1.4;
}

.principle-shape {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 3px 8px;
  border-radius: 5px;
  flex-shrink: 0;
}

.principle-body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-bottom: 14px;
}

.principle-col {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.principle-col p {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-light);
  margin: 0;
}

.col-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--color-primary);
  text-transform: uppercase;
}

.principle-col:nth-child(2) .col-label {
  color: #FF6B9D;
}

.principle-code {
  display: block;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12.5px;
  color: var(--color-text);
  background: var(--color-bg);
  padding: 8px 12px;
  border-radius: 6px;
  overflow-x: auto;
  white-space: nowrap;
}

.principle-example {
  margin-top: 10px;
  border-radius: 8px;
  background: rgba(108, 99, 255, 0.04);
  border: 1px solid rgba(108, 99, 255, 0.15);
  overflow: hidden;
}

.principle-example summary {
  padding: 8px 14px;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--color-primary);
  cursor: pointer;
  user-select: none;
  list-style: none;
  transition: background 0.15s;
}

.principle-example summary::-webkit-details-marker {
  display: none;
}

.principle-example summary:hover {
  background: rgba(108, 99, 255, 0.08);
}

.principle-example[open] summary {
  border-bottom: 1px solid rgba(108, 99, 255, 0.15);
  background: rgba(108, 99, 255, 0.08);
}

.weights-section {
  margin-bottom: 60px;
}

.weights-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
  margin-bottom: 24px;
}

.weight-card {
  background: var(--color-card);
  border-radius: 12px;
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
  transition: var(--transition);
}

.weight-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.weight-head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 8px;
}

.weight-name {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 18px;
  font-weight: 800;
  color: var(--color-primary);
}

.weight-shape {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 2px 8px;
  border-radius: 5px;
}

.weight-role {
  font-size: 13px;
  line-height: 1.65;
  color: var(--color-text-light);
  margin: 0 0 12px 0;
}

.weight-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  padding-top: 12px;
  border-top: 1px dashed rgba(0, 0, 0, 0.08);
}

.weight-stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: var(--color-text-light);
  text-transform: uppercase;
}

.stat-value {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  font-weight: 700;
  color: var(--color-text);
}

.weights-notes {
  background: rgba(0, 210, 255, 0.05);
  border-left: 3px solid #00D2FF;
  border-radius: 10px;
  padding: 18px 22px;
}

.notes-title {
  font-size: 14px;
  font-weight: 700;
  margin: 0 0 12px 0;
  color: var(--color-text);
}

.weights-notes ul {
  margin: 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.weights-notes li {
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--color-text-light);
}

.weights-notes code {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  background: var(--color-bg);
  padding: 1px 6px;
  border-radius: 4px;
  color: var(--color-text);
}

.weights-notes a {
  color: var(--color-primary);
  text-decoration: none;
  font-weight: 600;
}

.weights-notes a:hover {
  text-decoration: underline;
}

.principle-example pre {
  margin: 0;
  padding: 12px 14px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  line-height: 1.7;
  color: var(--color-text);
  background: transparent;
  white-space: pre;
  overflow-x: auto;
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

@media (max-width: 768px) {
  .features-grid {
    grid-template-columns: 1fr;
  }

  .intuition-grid {
    grid-template-columns: 1fr;
  }

  .weights-grid {
    grid-template-columns: 1fr;
  }

  .principle-body {
    grid-template-columns: 1fr;
    gap: 12px;
  }

  .title-text {
    font-size: 36px;
  }
}
</style>
