<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import ProbabilityChart from '../components/ProbabilityChart.vue'
import NetworkVisual from '../components/NetworkVisual.vue'
import ComputationTrace from '../components/ComputationTrace.vue'
import { MnistInference } from '../inference.js'

const inference = new MnistInference()
const COLLECTION_STORAGE_KEY = 'ctensor-playground-collection-v1'
const DIGIT_OPTIONS = Array.from({ length: 10 }, (_, i) => i)
const MODEL = {
  key: 'mnist',
  name: 'MNIST Minis',
  url: './weights.bin',
  description: '线上线下统一使用轻量 MNIST 模型，减少 GitHub Pages 首次加载等待。',
}
const modelReady = ref(false)
const modelError = ref(null)
let modelLoadVersion = 0

const currentModel = computed(() => MODEL)

const handleCollectorKeydown = (e) => {
  if (!result.value) return
  if (e.metaKey || e.ctrlKey || e.altKey) return

  if (/^[0-9]$/.test(e.key)) {
    selectCollectorLabel(e.key)
    e.preventDefault()
    return
  }

  if (e.key === 'Enter' && collectorLabel.value !== '') {
    saveCollectedLabel(collectorLabel.value)
    e.preventDefault()
  }
}

onMounted(async () => {
  await loadSelectedModel(false)
  setupCanvas()
  await refreshCollectedCount()
  window.addEventListener('keydown', handleCollectorKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleCollectorKeydown)
})

const canvasRef = ref(null)
const pixelGridRef = ref(null)
const isDrawing = ref(false)
const isAnalyzing = ref(false)
const result = ref(null)
const hasDrawn = ref(false)
const currentStep = ref(-1)
const hoveredPixel = ref(null)
const collectorLabel = ref('')
const collectorSaved = ref(false)
const collectorStatus = ref('')
const collectedCount = ref(0)
const collectorMode = ref('local')
const collectorPathHint = ref('')

const strokeHistory = reactive([])
let currentStroke = []

const canvasSize = 280
const gridSize = 28

const computeSteps = [
  { id: 'input', label: 'Step 1: 输入', code: 'x [1, 784]', desc: '28×28 像素展平为一维向量，值域 [-1, 1]', detail: '每个像素从 canvas 的灰度值 (0-255) 归一化到 [-1, 1]。黑色=1，白色=-1。' },
  { id: 'matmul1', label: 'Step 2: 矩阵乘法', code: 'h1 = x @ W1  →  [1, 256]', desc: '[1,784] × [784,256] = [1,256]', detail: 'C[i,j] = Σ x[0,k] × W1[k,j]，共 784×256 = 200,704 次乘加运算。macOS 上使用 cblas_sgemm (Accelerate 框架) 加速。' },
  { id: 'bias1', label: 'Step 3: 加偏置', code: 'h1b = h1 + b1  →  [1, 256]', desc: '每个神经元加一个偏置值', detail: 'h1b[0,i] = h1[0,i] + b1[i]。偏置的作用是让神经元有一个基础激活阈值，类似神经元的"静止电位"。' },
  { id: 'relu', label: 'Step 4: ReLU 激活', code: 'r1 = max(0, h1b)  →  [1, 256]', desc: '负值归零，正值保留', detail: 'ReLU(x) = max(0, x)。这是网络的"非线性"来源——没有激活函数的话，多层矩阵乘法等价于一层。ReLU 让网络能学习复杂的模式。' },
  { id: 'matmul2', label: 'Step 5: 矩阵乘法', code: 'h2 = r1 @ W2  →  [1, 10]', desc: '[1,256] × [256,10] = [1,10]', detail: '第二次矩阵乘法，将 256 维的隐藏表示映射到 10 个类别的分数。每个输出维度对应一个数字 (0-9) 的"匹配度"。' },
  { id: 'bias2', label: 'Step 6: 加偏置', code: 'h2b = h2 + b2  →  [1, 10]', desc: '每个输出加偏置', detail: 'h2b[0,i] = h2[0,i] + b2[i]。10 个偏置值，每个对应一个数字类别。' },
  { id: 'logsoftmax', label: 'Step 7: LogSoftmax', code: 'out = log_softmax(h2b)', desc: '将分数转为对数概率分布', detail: 'log_softmax(x)[i] = x[i] - max(x) - log(Σ exp(x[j] - max(x)))。减 max 是为了数值稳定性。结果满足 Σ exp(out[i]) = 1。' },
  { id: 'argmax', label: 'Step 8: Argmax', code: 'predicted = argmax(out)', desc: '取最大概率对应的数字', detail: '遍历 10 个输出值，找到最大的那个的索引。这就是网络的最终预测。置信度 = exp(out[predicted])。' },
]

function setupCanvas() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, canvasSize, canvasSize)
  ctx.strokeStyle = 'white'
  ctx.lineWidth = 16
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  canvas.addEventListener('mousedown', startDrawing)
  canvas.addEventListener('mousemove', draw)
  canvas.addEventListener('mouseup', stopDrawing)
  canvas.addEventListener('mouseleave', stopDrawing)
  canvas.addEventListener('touchstart', handleTouch, { passive: false })
  canvas.addEventListener('touchmove', handleTouchMove, { passive: false })
  canvas.addEventListener('touchend', stopDrawing)
}

function getPos(e) {
  const canvas = canvasRef.value
  const rect = canvas.getBoundingClientRect()
  const scaleX = canvasSize / rect.width
  const scaleY = canvasSize / rect.height
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  }
}

function startDrawing(e) {
  isDrawing.value = true
  currentStroke = []
  const pos = getPos(e)
  currentStroke.push(pos)
  const ctx = canvasRef.value.getContext('2d')
  ctx.beginPath()
  ctx.moveTo(pos.x, pos.y)
}

function draw(e) {
  if (!isDrawing.value) return
  const pos = getPos(e)
  currentStroke.push(pos)
  const ctx = canvasRef.value.getContext('2d')
  ctx.lineTo(pos.x, pos.y)
  ctx.stroke()
}

function stopDrawing() {
  if (isDrawing.value) {
    isDrawing.value = false
    if (currentStroke.length > 0) {
      strokeHistory.push([...currentStroke])
    }
    hasDrawn.value = true
  }
}

function handleTouch(e) {
  e.preventDefault()
  const touch = e.touches[0]
  const mouseEvent = new MouseEvent('mousedown', {
    clientX: touch.clientX,
    clientY: touch.clientY,
  })
  canvasRef.value.dispatchEvent(mouseEvent)
}

function handleTouchMove(e) {
  e.preventDefault()
  const touch = e.touches[0]
  const mouseEvent = new MouseEvent('mousemove', {
    clientX: touch.clientX,
    clientY: touch.clientY,
  })
  canvasRef.value.dispatchEvent(mouseEvent)
}

function clearCanvas() {
  const ctx = canvasRef.value.getContext('2d')
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, canvasSize, canvasSize)
  strokeHistory.length = 0
  currentStroke = []
  result.value = null
  hasDrawn.value = false
  currentStep.value = -1
  resetCollectorState()
}

function undoStroke() {
  if (strokeHistory.length === 0) return
  strokeHistory.pop()
  redrawCanvas()
  if (strokeHistory.length === 0) {
    result.value = null
    hasDrawn.value = false
    currentStep.value = -1
    resetCollectorState()
  }
}

function redrawCanvas() {
  const ctx = canvasRef.value.getContext('2d')
  ctx.fillStyle = 'black'
  ctx.fillRect(0, 0, canvasSize, canvasSize)
  ctx.strokeStyle = 'white'
  ctx.lineWidth = 16
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  for (const stroke of strokeHistory) {
    if (stroke.length === 0) continue
    ctx.beginPath()
    ctx.moveTo(stroke[0].x, stroke[0].y)
    for (let i = 1; i < stroke.length; i++) {
      ctx.lineTo(stroke[i].x, stroke[i].y)
    }
    ctx.stroke()
  }
}

function getPixelData() {
  const canvas = canvasRef.value
  const ctx = canvas.getContext('2d')

  // 1. 获取原始像素，找到数字的 bounding box
  const srcData = ctx.getImageData(0, 0, canvasSize, canvasSize)
  let minX = canvasSize, minY = canvasSize, maxX = 0, maxY = 0
  for (let y = 0; y < canvasSize; y++) {
    for (let x = 0; x < canvasSize; x++) {
      if (srcData.data[(y * canvasSize + x) * 4] > 10) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
  }

  // 没有笔画时返回全黑
  if (maxX <= minX || maxY <= minY) {
    return new Array(gridSize * gridSize).fill(-1)
  }

  // 2. 按 bounding box 裁剪，保持宽高比缩放到 20x20 区域，居中放置到 28x28
  const bw = maxX - minX + 1
  const bh = maxY - minY + 1
  const targetSize = 18  // 数字占 18x18 区域（比 MNIST 标准 20x20 略小，给质心居中留出更多空间，避免 offset clamp 破坏居中）
  const scale = targetSize / Math.max(bw, bh)
  const scaledW = Math.round(bw * scale)
  const scaledH = Math.round(bh * scale)

  // 计算质心用于居中
  let cx = 0, cy = 0, total = 0
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const v = srcData.data[(y * canvasSize + x) * 4]
      if (v > 0) {
        cx += x * v
        cy += y * v
        total += v
      }
    }
  }
  cx = (cx / total - minX) * scale
  cy = (cy / total - minY) * scale

  // 偏移量：让质心在 28x28 的中心 (14, 14)，并 clamp 防止裁剪
  let offsetX = Math.round(14 - cx)
  let offsetY = Math.round(14 - cy)
  offsetX = Math.max(0, Math.min(gridSize - scaledW, offsetX))
  offsetY = Math.max(0, Math.min(gridSize - scaledH, offsetY))

  // 用一个较大的中间画布，先缩放再模糊，最后提取 28x28
  // 这样能产生类似 MNIST 的平滑笔画
  const upscale = 4  // 先画到 112x112 再缩到 28x28
  const midSize = gridSize * upscale
  const midCanvas = document.createElement('canvas')
  midCanvas.width = midSize
  midCanvas.height = midSize
  const midCtx = midCanvas.getContext('2d')

  midCtx.fillStyle = 'black'
  midCtx.fillRect(0, 0, midSize, midSize)
  midCtx.imageSmoothingEnabled = true
  midCtx.imageSmoothingQuality = 'high'

  // 将裁剪区域缩放后居中绘制到中间画布
  midCtx.drawImage(
    canvas,
    minX, minY, bw, bh,
    offsetX * upscale, offsetY * upscale, scaledW * upscale, scaledH * upscale
  )

  // 在中间画布上应用高斯模糊，让笔画更柔和（模拟 MNIST 的抗锯齿效果）
  midCtx.filter = 'blur(1px)'
  midCtx.drawImage(midCanvas, 0, 0)
  midCtx.filter = 'none'

  // 最终缩放到 28x28
  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = gridSize
  tempCanvas.height = gridSize
  const tempCtx = tempCanvas.getContext('2d')
  tempCtx.imageSmoothingEnabled = true
  tempCtx.imageSmoothingQuality = 'high'
  tempCtx.drawImage(midCanvas, 0, 0, gridSize, gridSize)

  // 3. 提取像素并归一化到 [-1, 1]
  const imageData = tempCtx.getImageData(0, 0, gridSize, gridSize)
  const pixels = []
  for (let i = 0; i < imageData.data.length; i += 4) {
    const gray = imageData.data[i]
    const normalized = (gray / 255) * 2 - 1
    pixels.push(normalized)
  }
  return pixels
}

function drawPixelGrid() {
  const canvas = pixelGridRef.value
  if (!canvas || !result.value) return
  const ctx = canvas.getContext('2d')
  const pixels = result.value.pixels
  const cellW = canvas.width / gridSize
  const cellH = canvas.height / gridSize

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const v = (pixels[y * gridSize + x] + 1) / 2
      const brightness = Math.round(v * 255)
      ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`
      ctx.fillRect(x * cellW, y * cellH, cellW - 0.5, cellH - 0.5)
    }
  }

  if (hoveredPixel.value) {
    ctx.strokeStyle = '#6C63FF'
    ctx.lineWidth = 2
    ctx.strokeRect(hoveredPixel.value.col * cellW, hoveredPixel.value.row * cellH, cellW, cellH)
  }
}

watch(hoveredPixel, () => { drawPixelGrid() })

function handlePixelGridHover(e) {
  const canvas = pixelGridRef.value
  if (!canvas || !result.value) return
  const rect = canvas.getBoundingClientRect()
  const scaleX = canvas.width / rect.width
  const scaleY = canvas.height / rect.height
  const x = Math.floor((e.clientX - rect.left) * scaleX / (canvas.width / gridSize))
  const y = Math.floor((e.clientY - rect.top) * scaleY / (canvas.height / gridSize))
  if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
    const idx = y * gridSize + x
    const val = result.value.pixels[idx]
    hoveredPixel.value = { row: y, col: x, idx, val }
  } else {
    hoveredPixel.value = null
  }
}

function handlePixelGridLeave() {
  hoveredPixel.value = null
}

// Step visualization refs
const stepPixelRef = ref(null)
const preReluRef = ref(null)
const reluBeforeRef = ref(null)
const reluAfterRef = ref(null)
const matmul2InputRef = ref(null)

const pixelStats = computed(() => {
  if (!result.value) return null
  const pixels = result.value.pixels
  const nonZero = pixels.filter(p => p > -0.9).length
  return { nonZero, total: 784 }
})

const reluStats = computed(() => {
  if (!result.value?.hidden) return null
  const active = result.value.hidden.filter(v => v > 0).length
  return { active, dead: 256 - active }
})

const preSoftmaxBars = computed(() => {
  if (!result.value?.pre_softmax) return []
  const vals = result.value.pre_softmax
  const maxAbs = Math.max(...vals.map(Math.abs), 0.001)
  return vals.map((v, i) => ({
    digit: i,
    value: v,
    width: (Math.abs(v) / maxAbs) * 100,
    positive: v > 0,
  }))
})

function drawPixelsOnCanvas(canvas, pixels) {
  if (!canvas || !pixels) return
  const ctx = canvas.getContext('2d')
  const cellW = canvas.width / 28
  const cellH = canvas.height / 28
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = (pixels[y * 28 + x] + 1) / 2
      const b = Math.round(v * 255)
      ctx.fillStyle = `rgb(${b}, ${b}, ${b})`
      ctx.fillRect(x * cellW, y * cellH, cellW - 0.5, cellH - 0.5)
    }
  }
}

function drawHeatmapOnCanvas(canvas, data, cols) {
  if (!canvas || !data || !data.length) return
  const ctx = canvas.getContext('2d')
  const rows = Math.ceil(data.length / cols)
  const cellW = canvas.width / cols
  const cellH = canvas.height / rows
  const maxVal = Math.max(...data.map(Math.abs), 0.001)

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  data.forEach((val, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    const norm = Math.abs(val) / maxVal
    const r = val > 0 ? Math.round(108 * norm) : Math.round(255 * norm)
    const g = val > 0 ? Math.round(99 * norm) : Math.round(107 * norm)
    const b = val > 0 ? Math.round(255 * norm) : Math.round(157 * norm)
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
    ctx.fillRect(col * cellW + 0.5, row * cellH + 0.5, cellW - 1, cellH - 1)
  })
}

function drawStepVisual() {
  if (!result.value) return
  const step = currentStep.value

  if (step === 0) {
    drawPixelsOnCanvas(stepPixelRef.value, result.value.pixels)
  } else if (step === 1 || step === 2) {
    drawPixelsOnCanvas(stepPixelRef.value, result.value.pixels)
    drawHeatmapOnCanvas(preReluRef.value, result.value.pre_relu, 16)
  } else if (step === 3) {
    drawHeatmapOnCanvas(reluBeforeRef.value, result.value.pre_relu, 16)
    drawHeatmapOnCanvas(reluAfterRef.value, result.value.hidden, 16)
  } else if (step === 4) {
    drawHeatmapOnCanvas(matmul2InputRef.value, result.value.hidden, 16)
  }
}

watch([() => currentStep.value, () => result.value], () => {
  nextTick(() => drawStepVisual())
}, { flush: 'post' })

const apiError = ref(null)

function resetCollectorState() {
  collectorLabel.value = ''
  collectorSaved.value = false
  collectorStatus.value = ''
}

function normalizeCollectorLabel(value) {
  const label = Number(value)
  if (!Number.isInteger(label) || label < 0 || label > 9) return null
  return label
}

function selectCollectorLabel(label) {
  collectorLabel.value = String(label)
  if (!collectorSaved.value) {
    collectorStatus.value = ''
  }
}

function confirmSelectedCollectorLabel() {
  saveCollectedLabel(collectorLabel.value)
}

function loadCollectedSamples() {
  try {
    const raw = localStorage.getItem(COLLECTION_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch (e) {
    console.error('Failed to load collected samples:', e)
    return []
  }
}

function saveCollectedSamples(samples) {
  localStorage.setItem(COLLECTION_STORAGE_KEY, JSON.stringify(samples))
}

async function refreshCollectedCount() {
  try {
    const stats = await fetchCollectorStats()
    collectorMode.value = 'filesystem'
    collectedCount.value = stats.count || 0
    collectorPathHint.value = stats.paths?.csv || ''
  } catch {
    collectorMode.value = 'local'
    collectedCount.value = loadCollectedSamples().length
    collectorPathHint.value = ''
  }
}

function makeOneHot(label) {
  const onehot = new Array(10).fill(0)
  onehot[label] = 1
  return onehot
}

function buildCollectedSample(label) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    createdAt: new Date().toISOString(),
    label,
    predicted: result.value.predicted,
    confidence: result.value.confidence,
    model: currentModel.value.key,
    pixels: [...result.value.pixels],
    onehot: makeOneHot(label),
    strokeHistory: strokeHistory.map(stroke => stroke.map(point => ({ x: point.x, y: point.y }))),
  }
}

function downloadTextFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

async function fetchCollectorStats() {
  const response = await fetch('/api/collect-digit/stats')
  if (!response.ok) {
    throw new Error(`stats request failed: ${response.status}`)
  }
  return response.json()
}

async function saveSampleToFilesystem(sample) {
  const response = await fetch('/api/collect-digit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sample),
  })
  if (!response.ok) {
    throw new Error(`save request failed: ${response.status}`)
  }
  return response.json()
}

async function clearFilesystemSamples() {
  const response = await fetch('/api/collect-digit', {
    method: 'DELETE',
  })
  if (!response.ok) {
    throw new Error(`clear request failed: ${response.status}`)
  }
  return response.json()
}

function triggerApiDownload(format) {
  const link = document.createElement('a')
  link.href = `/api/collect-digit/export?format=${format}`
  link.download = format === 'csv' ? 'playground_collection.csv' : 'playground_collection.jsonl'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

async function saveCollectedLabel(labelValue) {
  if (!result.value) {
    collectorStatus.value = '请先让 AI 完成一次预测。'
    return
  }
  if (collectorSaved.value) {
    collectorStatus.value = '这次样本已经保存过了，重新书写后再保存下一条。'
    return
  }

  const label = normalizeCollectorLabel(labelValue)
  if (label === null) {
    collectorStatus.value = '正确数字只能是 0-9。'
    return
  }

  const sample = buildCollectedSample(label)

  try {
    const payload = await saveSampleToFilesystem(sample)
    collectorMode.value = 'filesystem'
    collectedCount.value = payload.count || 0
    collectorPathHint.value = payload.paths?.csv || ''
    collectorSaved.value = true
    collectorLabel.value = String(label)
    collectorStatus.value = collectorPathHint.value
      ? `已写入项目目录 ${collectorPathHint.value}，当前共 ${collectedCount.value} 条样本。`
      : `已写入项目目录，当前共 ${collectedCount.value} 条样本。`
    return
  } catch (e) {
    console.warn('Failed to save to filesystem, falling back to localStorage:', e)
  }

  collectorMode.value = 'local'
  const samples = loadCollectedSamples()
  samples.push(sample)
  saveCollectedSamples(samples)
  collectedCount.value = samples.length
  collectorSaved.value = true
  collectorLabel.value = String(label)
  collectorStatus.value = `已保存到浏览器本地，当前共 ${samples.length} 条样本。`
}

async function exportCollectedJson() {
  if (collectorMode.value === 'filesystem') {
    triggerApiDownload('jsonl')
    collectorStatus.value = '已从项目目录导出 JSONL 样本。'
    return
  }

  const samples = loadCollectedSamples()
  if (samples.length === 0) {
    collectorStatus.value = '当前还没有可导出的采集样本。'
    return
  }
  downloadTextFile(
    `playground_collection_${samples.length}.json`,
    JSON.stringify(samples, null, 2),
    'application/json'
  )
  collectorStatus.value = `已导出 ${samples.length} 条 JSON 样本。`
}

async function exportCollectedCsv() {
  if (collectorMode.value === 'filesystem') {
    triggerApiDownload('csv')
    collectorStatus.value = '已从项目目录导出 CSV 样本。'
    return
  }

  const samples = loadCollectedSamples()
  if (samples.length === 0) {
    collectorStatus.value = '当前还没有可导出的采集样本。'
    return
  }

  const rows = samples.map(sample => [...sample.pixels, ...makeOneHot(sample.label)].join(','))
  downloadTextFile(
    `playground_collection_${samples.length}.csv`,
    rows.join('\n'),
    'text/csv;charset=utf-8'
  )
  collectorStatus.value = `已导出 ${samples.length} 条 CSV 样本。`
}

async function clearCollectedSamples() {
  if (collectorMode.value === 'filesystem') {
    try {
      await clearFilesystemSamples()
      collectedCount.value = 0
      collectorStatus.value = '已清空项目目录中的采集数据。'
      return
    } catch (e) {
      console.warn('Failed to clear filesystem samples, falling back to localStorage:', e)
    }
  }

  localStorage.removeItem(COLLECTION_STORAGE_KEY)
  collectorMode.value = 'local'
  collectedCount.value = 0
  collectorPathHint.value = ''
  collectorStatus.value = '已清空浏览器本地采集数据。'
}

async function loadSelectedModel(rerun = true) {
  const version = ++modelLoadVersion
  modelReady.value = false
  modelError.value = null
  apiError.value = null

  try {
    await inference.loadWeights(currentModel.value.url)
    if (version !== modelLoadVersion) return
    modelReady.value = true

    if (rerun && hasDrawn.value) {
      runPrediction(getPixelData())
    }
  } catch (e) {
    if (version !== modelLoadVersion) return
    modelError.value = e.message
    result.value = null
    currentStep.value = -1
    console.error('Failed to load model weights:', e)
  }
}

function runPrediction(pixels) {
  result.value = inference.predict(pixels)
  collectorLabel.value = String(result.value.predicted)
  collectorSaved.value = false
  collectorStatus.value = ''
  console.log('Inference result:', {
    model: currentModel.value.key,
    predicted: result.value.predicted,
    confidence: result.value.confidence,
  })
  currentStep.value = computeSteps.length - 1
  drawPixelGrid()
}

async function analyze() {
  if (!hasDrawn.value) return
  if (!modelReady.value) {
    apiError.value = `模型权重未加载，请检查 ${currentModel.value.url} 是否存在`
    return
  }
  isAnalyzing.value = true
  currentStep.value = 0
  apiError.value = null

  const pixels = getPixelData()

  try {
    runPrediction(pixels)
  } catch (e) {
    console.error('Inference error:', e)
    apiError.value = e.message
  } finally {
    isAnalyzing.value = false
  }
}

function stepForward() {
  if (currentStep.value < computeSteps.length - 1) {
    currentStep.value++
  }
}

function stepBack() {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

const currentStepData = computed(() => {
  if (currentStep.value < 0 || currentStep.value >= computeSteps.length) return null
  return computeSteps[currentStep.value]
})

const networkHint = computed(() => {
  const hints = [
    '当前：数据进入输入层 (784 个像素值)',
    '当前：输入层 → 隐藏层，200,704 个权重参与计算',
    '当前：隐藏层 256 个神经元加上偏置',
    '当前：ReLU 激活，部分神经元被关闭',
    '当前：隐藏层 → 输出层，2,560 个权重参与计算',
    '当前：输出层 10 个神经元加上偏置',
    '当前：LogSoftmax 将分数转为概率',
    '当前：取概率最大的数字作为预测结果',
  ]
  return hints[currentStep.value] || ''
})

const progressPercent = computed(() => {
  if (currentStep.value < 0) return 0
  return ((currentStep.value + 1) / computeSteps.length) * 100
})
</script>

<template>
  <div class="playground-view">
    <header class="page-header">
      <h1>📐 矩阵计算演示</h1>
      <p class="page-desc">手写一个数字，逐步查看 784→256→10 的完整计算过程</p>
      <div class="model-switcher static">
        <span class="model-label">固定模型</span>
        <div class="model-pill">{{ currentModel.name }}</div>
        <p class="model-description">{{ currentModel.description }}</p>
      </div>
    </header>

    <div class="main-layout">
      <div class="left-panel">
        <div class="canvas-section">
          <h3 class="section-label">手写输入</h3>
          <div class="canvas-wrapper">
            <canvas
              ref="canvasRef"
              :width="canvasSize"
              :height="canvasSize"
              class="draw-canvas"
            ></canvas>
          </div>
          <div class="canvas-controls">
            <button class="control-btn" @click="undoStroke" :disabled="strokeHistory.length === 0">
              ↩ 撤销
            </button>
            <button class="control-btn" @click="clearCanvas">
              ✕ 清空
            </button>
            <button class="control-btn analyze-btn" @click="analyze" :disabled="!hasDrawn || isAnalyzing">
              {{ isAnalyzing ? '计算中...' : '▶ 执行前向传播' }}
            </button>
          </div>
        </div>

        <div class="pixel-grid-section">
          <h3 class="section-label">28×28 像素矩阵 [1, 784]</h3>
          <div class="pixel-grid-wrapper">
            <canvas
              ref="pixelGridRef"
              width="224"
              height="224"
              class="pixel-grid"
              @mousemove="handlePixelGridHover"
              @mouseleave="handlePixelGridLeave"
            ></canvas>
            <div v-if="hoveredPixel" class="pixel-tooltip">
              <span class="pixel-tooltip-coord">({{ hoveredPixel.row }}, {{ hoveredPixel.col }})</span>
              <span class="pixel-tooltip-idx">索引: {{ hoveredPixel.idx }}</span>
              <span class="pixel-tooltip-val">值: {{ hoveredPixel.val.toFixed(4) }}</span>
            </div>
          </div>
          <p class="grid-hint">黑色 = -1（背景），白色 = 1（笔画）</p>
        </div>
      </div>

      <div class="right-panel">
        <div v-if="modelError" class="error-display">
          <span class="error-icon">⚠️</span>
          <h3>模型加载失败</h3>
          <p class="error-message">{{ modelError }}</p>
          <p class="error-hint">请确保 <code>{{ currentModel.url }}</code> 存在于 <code>web-app/public/</code> 目录下</p>
        </div>

        <div v-if="!modelReady && !modelError" class="result-placeholder">
          <span class="placeholder-icon">⏳</span>
          <p>加载模型权重中...</p>
        </div>

        <div v-if="!result && !apiError && modelReady" class="result-placeholder">
          <span class="placeholder-icon">←</span>
          <p>在左侧画一个数字，然后点击"执行前向传播"</p>
        </div>

        <div v-if="apiError" class="error-display">
          <span class="error-icon">⚠️</span>
          <h3>推理失败</h3>
          <p class="error-message">{{ apiError }}</p>
        </div>

        <template v-if="result">
          <div class="collector-panel">
            <div class="collector-header">
              <div>
                <h3>真实标签采集</h3>
                <p>AI 出结果后，你可以直接确认预测，或者点选正确数字再保存。</p>
                <p class="collector-storage-hint">
                  <template v-if="collectorMode === 'filesystem'">
                    本地开发模式：样本会自动写入项目目录
                    <code v-if="collectorPathHint">{{ collectorPathHint }}</code>
                  </template>
                  <template v-else>
                    当前未连接到本地文件保存接口，会回退到浏览器本地保存。
                  </template>
                </p>
              </div>
              <div class="collector-count">已采集 {{ collectedCount }}</div>
            </div>

            <div class="collector-main">
              <div class="collector-prediction">
                <span class="collector-pred-label">AI 预测</span>
                <span class="collector-pred-value">{{ result.predicted }}</span>
                <span class="collector-pred-confidence">{{ (result.confidence * 100).toFixed(1) }}%</span>
              </div>

              <div class="collector-actions">
                <button class="collector-btn primary" @click="saveCollectedLabel(result.predicted)">
                  预测正确，保存
                </button>
                <div class="collector-corrector">
                  <span class="collector-corrector-label">如果不对，点选正确数字</span>
                  <div class="collector-digit-grid">
                    <button
                      v-for="digit in DIGIT_OPTIONS"
                      :key="digit"
                      :class="['collector-digit-btn', { active: collectorLabel === String(digit) }]"
                      @click="selectCollectorLabel(digit)"
                    >
                      {{ digit }}
                    </button>
                  </div>
                  <div class="collector-confirm-row">
                    <button class="collector-btn" @click="confirmSelectedCollectorLabel">
                      保存所选标签
                    </button>
                    <span class="collector-hint">键盘可直接按数字选择，按 Enter 保存</span>
                  </div>
                  <button
                    v-if="collectorLabel === String(result.predicted)"
                    class="collector-btn subtle"
                    @click="saveCollectedLabel(result.predicted)"
                  >
                    当前选中就是 AI 预测值，直接保存
                  </button>
                </div>
              </div>
            </div>

            <div class="collector-export">
              <button class="collector-btn secondary" @click="exportCollectedJson">导出 JSON</button>
              <button class="collector-btn secondary" @click="exportCollectedCsv">导出 CSV</button>
              <button class="collector-btn danger" @click="clearCollectedSamples">清空本地数据</button>
            </div>

            <p v-if="collectorStatus" class="collector-status" :class="{ saved: collectorSaved }">
              {{ collectorStatus }}
            </p>
          </div>

          <div class="step-progress">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
            </div>
            <div class="step-nav">
              <button class="step-btn" @click="stepBack" :disabled="currentStep <= 0">← 上一步</button>
              <span class="step-counter">{{ currentStep + 1 }} / {{ computeSteps.length }}</span>
              <button class="step-btn" @click="stepForward" :disabled="currentStep >= computeSteps.length - 1">下一步 →</button>
            </div>
          </div>

          <div class="step-display">
            <div class="step-header">
              <h3 class="step-title">{{ currentStepData.label }}</h3>
              <code class="step-code">{{ currentStepData.code }}</code>
            </div>
            <p class="step-desc">{{ currentStepData.desc }}</p>
            <div class="step-detail">{{ currentStepData.detail }}</div>
          </div>

          <div class="step-pipeline">
            <div
              v-for="(step, i) in computeSteps"
              :key="step.id"
              :class="['pipeline-node', { active: i === currentStep, done: i < currentStep }]"
              @click="currentStep = i"
            >
              <span class="node-dot"></span>
              <span class="node-label">{{ step.label.split(': ')[1] }}</span>
            </div>
          </div>

          <!-- 每个步骤的可视化 -->
          <div class="step-visual">
            <!-- Step 0: 输入像素 -->
            <div v-if="currentStep === 0" class="visual-block">
              <h3 class="visual-title">输入像素矩阵</h3>
              <canvas ref="stepPixelRef" width="224" height="224" class="visual-pixel-canvas"></canvas>
              <div class="visual-stats" v-if="pixelStats">
                <span class="stat-item">总像素: <strong>{{ pixelStats.total }}</strong> (28×28)</span>
                <span class="stat-item">有效像素: <strong>{{ pixelStats.nonZero }}</strong></span>
                <span class="stat-item">值域: <strong>[-1, 1]</strong></span>
              </div>
            </div>

            <!-- Step 1: 矩阵乘法 -->
            <div v-if="currentStep === 1" class="visual-block">
              <h3 class="visual-title">矩阵乘法过程</h3>
              <div class="matmul-flow">
                <div class="flow-item">
                  <canvas ref="stepPixelRef" width="112" height="112" class="flow-canvas"></canvas>
                  <span class="flow-label">输入 x<br>[1, 784]</span>
                </div>
                <span class="flow-op">×</span>
                <div class="flow-item">
                  <div class="weight-placeholder">
                    <span class="weight-count">200,704</span>
                    <span class="weight-unit">个参数</span>
                  </div>
                  <span class="flow-label">权重 W1<br>[784, 256]</span>
                </div>
                <span class="flow-op">=</span>
                <div class="flow-item">
                  <canvas ref="preReluRef" width="160" height="160" class="flow-canvas heatmap-canvas"></canvas>
                  <span class="flow-label">输出 h1<br>[1, 256]</span>
                </div>
              </div>
            </div>

            <!-- Step 2: 加偏置 -->
            <div v-if="currentStep === 2" class="visual-block">
              <h3 class="visual-title">加偏置后的 256 个神经元值</h3>
              <p class="visual-subtitle">偏置让每个神经元有一个基础激活阈值，类似"静息电位"</p>
              <canvas ref="preReluRef" width="256" height="256" class="heatmap-canvas center-canvas"></canvas>
              <div class="heatmap-legend-inline">
                <span><span class="legend-dot positive"></span> 正值（将被 ReLU 保留）</span>
                <span><span class="legend-dot negative"></span> 负值（将被 ReLU 归零）</span>
              </div>
            </div>

            <!-- Step 3: ReLU 前后对比 -->
            <div v-if="currentStep === 3" class="visual-block">
              <h3 class="visual-title">ReLU 前后对比</h3>
              <div class="relu-comparison">
                <div class="relu-side">
                  <span class="relu-label">ReLU 前（有正有负）</span>
                  <canvas ref="reluBeforeRef" width="192" height="192" class="heatmap-canvas"></canvas>
                </div>
                <div class="relu-arrow">
                  <span>max(0, x)</span>
                  <span class="arrow-icon">→</span>
                </div>
                <div class="relu-side">
                  <span class="relu-label">ReLU 后（负值归零）</span>
                  <canvas ref="reluAfterRef" width="192" height="192" class="heatmap-canvas"></canvas>
                </div>
              </div>
              <div class="relu-stats" v-if="reluStats">
                <span class="stat-active">激活: <strong>{{ reluStats.active }}</strong> 个神经元</span>
                <span class="stat-dead">沉默: <strong>{{ reluStats.dead }}</strong> 个神经元 (输出=0)</span>
              </div>
            </div>

            <!-- Step 4: 第二层矩阵乘法 -->
            <div v-if="currentStep === 4" class="visual-block">
              <h3 class="visual-title">隐藏层 → 输出层</h3>
              <div class="matmul-flow">
                <div class="flow-item">
                  <canvas ref="matmul2InputRef" width="128" height="128" class="flow-canvas heatmap-canvas"></canvas>
                  <span class="flow-label">ReLU 输出<br>[1, 256]</span>
                </div>
                <span class="flow-op">×</span>
                <div class="flow-item">
                  <div class="weight-placeholder">
                    <span class="weight-count">2,560</span>
                    <span class="weight-unit">个参数</span>
                  </div>
                  <span class="flow-label">权重 W2<br>[256, 10]</span>
                </div>
                <span class="flow-op">=</span>
                <div class="flow-item">
                  <div class="output-bars-mini">
                    <div v-for="bar in preSoftmaxBars" :key="bar.digit" class="out-bar-row">
                      <span class="out-bar-digit">{{ bar.digit }}</span>
                      <div class="out-bar-track">
                        <div class="out-bar-fill" :class="{ positive: bar.positive, negative: !bar.positive }"
                          :style="{ width: bar.width + '%' }"></div>
                      </div>
                    </div>
                  </div>
                  <span class="flow-label">输出 h2<br>[1, 10]</span>
                </div>
              </div>
            </div>

            <!-- Step 5: 加偏置 -->
            <div v-if="currentStep === 5" class="visual-block">
              <h3 class="visual-title">10 个类别的原始分数</h3>
              <div class="score-bars">
                <div v-for="bar in preSoftmaxBars" :key="bar.digit" class="score-row">
                  <span class="score-digit">{{ bar.digit }}</span>
                  <div class="score-track">
                    <div class="score-fill" :class="{ positive: bar.positive, negative: !bar.positive }"
                      :style="{ width: bar.width + '%' }"></div>
                  </div>
                  <span class="score-value" :class="{ positive: bar.positive }">{{ bar.value.toFixed(2) }}</span>
                </div>
              </div>
              <p class="visual-hint">正值越大 → 越可能是该数字；负值 → 不太可能</p>
            </div>

            <!-- Step 6: LogSoftmax -->
            <div v-if="currentStep === 6" class="visual-block">
              <ProbabilityChart :output="result.output" />
            </div>

            <!-- Step 7: Argmax -->
            <div v-if="currentStep === 7" class="visual-block">
              <div class="final-prediction">
                <div class="pred-hero">
                  <span class="pred-big-number">{{ result.predicted }}</span>
                </div>
                <div class="pred-details">
                  <div class="confidence-row">
                    <span class="conf-label">置信度</span>
                    <div class="conf-bar-track">
                      <div class="conf-bar-fill" :style="{ width: (result.confidence * 100) + '%' }"></div>
                    </div>
                    <span class="conf-value">{{ (result.confidence * 100).toFixed(1) }}%</span>
                  </div>
                  <ProbabilityChart :output="result.output" />
                </div>
              </div>
            </div>
          </div>

          <!-- 真实计算过程展示 -->
          <ComputationTrace
            v-if="result && currentStep >= 0"
            :step="currentStep"
            :result="result"
            :weights="inference"
          />

          <!-- 神经网络连接图 -->
          <div class="network-block">
            <h3 class="visual-title">神经网络全局视图</h3>
            <p class="network-hint">{{ networkHint }}</p>
            <div class="network-wrapper">
              <NetworkVisual
                :step="currentStep"
                :hiddenActivations="result.hidden"
                :preRelu="result.pre_relu"
                :outputActivations="result.output"
                :preSoftmax="result.pre_softmax"
                :predicted="result.predicted"
              />
            </div>
          </div>
        </template>
      </div>
    </div>

    <div class="code-reference">
      <h3>对应的 C 代码</h3>
      <div class="code-block">
        <pre><code>// train.c 中的前向传播 (单样本)
Tensor *x  = tensor_from_arr(batch_x);     <span class="comment">// [1, 784] 输入</span>
Tensor *h1 = matmul(x, w1);                <span class="comment">// [1, 784] × [784, 256] → [1, 256]</span>
Tensor *h1b = add_bias(h1, b1);            <span class="comment">// + bias [256]</span>
Tensor *r1 = relu(h1b);                    <span class="comment">// ReLU 激活</span>
Tensor *h2 = matmul(r1, w2);               <span class="comment">// [1, 256] × [256, 10] → [1, 10]</span>
Tensor *h2b = add_bias(h2, b2);            <span class="comment">// + bias [10]</span>
Tensor *out = logsoftmax(h2b);             <span class="comment">// 对数概率分布</span>
int predicted = argmax(out->data->values); <span class="comment">// 取最大值的索引</span></code></pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.playground-view {
  max-width: 1300px;
  margin: 0 auto;
  padding: 40px 24px;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.model-switcher {
  margin: 18px auto 0;
  max-width: 560px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.model-switcher.static {
  padding: 14px 18px;
  border-radius: 16px;
  background: rgba(108, 99, 255, 0.06);
  border: 1px solid rgba(108, 99, 255, 0.14);
}

.model-label {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--color-text-light);
}

.model-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 8px 14px;
  border-radius: 999px;
  background: var(--color-card);
  color: var(--color-primary);
  font-size: 14px;
  font-weight: 800;
  box-shadow: var(--shadow-sm);
}

.model-description {
  margin: 0;
  max-width: 560px;
  font-size: 14px;
  color: var(--color-text-light);
}

.page-header h1 {
  font-size: 32px;
  font-weight: 800;
  margin-bottom: 8px;
}

.page-desc {
  font-size: 16px;
  color: var(--color-text-light);
}

.main-layout {
  display: grid;
  grid-template-columns: 380px 1fr;
  gap: 28px;
  margin-bottom: 40px;
}

.left-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.section-label {
  font-size: 14px;
  font-weight: 700;
  color: var(--color-text-light);
  margin-bottom: 10px;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.canvas-wrapper {
  background: var(--color-card);
  border-radius: 14px;
  padding: 12px;
  box-shadow: var(--shadow-sm);
}

.draw-canvas {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 10px;
  cursor: crosshair;
  touch-action: none;
  background: black;
}

.canvas-controls {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.control-btn {
  flex: 1;
  padding: 12px;
  border-radius: 10px;
  font-weight: 600;
  font-size: 14px;
  background: var(--color-bg);
  color: var(--color-text);
}

.control-btn:hover {
  background: var(--color-border);
}

.control-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.analyze-btn {
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  color: white;
}

.analyze-btn:hover:not(:disabled) {
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}

.pixel-grid-section {
  background: var(--color-card);
  border-radius: 14px;
  padding: 16px;
  box-shadow: var(--shadow-sm);
}

.pixel-grid-wrapper {
  position: relative;
  display: inline-block;
  width: 100%;
}

.pixel-grid {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 8px;
  image-rendering: pixelated;
  border: 1px solid var(--color-border);
  cursor: crosshair;
}

.pixel-tooltip {
  position: absolute;
  bottom: 8px;
  left: 8px;
  background: rgba(30, 30, 46, 0.92);
  color: #cdd6f4;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 12px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  gap: 2px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(4px);
}

.pixel-tooltip-coord {
  color: #89b4fa;
  font-weight: 600;
}

.pixel-tooltip-idx {
  color: #a6e3a1;
}

.pixel-tooltip-val {
  color: #f9e2af;
  font-weight: 700;
}

.grid-hint {
  font-size: 12px;
  color: var(--color-text-light);
  text-align: center;
  margin-top: 6px;
}

.right-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.collector-panel {
  background: var(--color-card);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
  border: 1px solid rgba(108, 99, 255, 0.12);
}

.collector-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.collector-header h3 {
  margin: 0 0 6px;
  font-size: 18px;
  font-weight: 800;
}

.collector-header p {
  margin: 0;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-text-light);
}

.collector-storage-hint {
  margin-top: 8px !important;
}

.collector-storage-hint code {
  margin-left: 6px;
  padding: 2px 6px;
  border-radius: 6px;
  background: var(--color-bg);
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
}

.collector-count {
  flex-shrink: 0;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(108, 99, 255, 0.1);
  color: var(--color-primary);
  font-size: 13px;
  font-weight: 800;
}

.collector-main {
  display: flex;
  gap: 18px;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}

.collector-prediction {
  min-width: 132px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 14px 16px;
  border-radius: 12px;
  background: var(--color-bg);
  text-align: center;
}

.collector-pred-label {
  font-size: 12px;
  color: var(--color-text-light);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.collector-pred-value {
  font-size: 42px;
  font-weight: 900;
  line-height: 1;
  color: var(--color-primary);
}

.collector-pred-confidence {
  font-size: 13px;
  font-weight: 700;
  color: var(--color-text-light);
}

.collector-actions {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.collector-corrector {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 10px;
  flex-wrap: wrap;
}

.collector-corrector-label {
  font-size: 13px;
  font-weight: 700;
  color: var(--color-text-light);
}

.collector-digit-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
  width: 100%;
  max-width: 320px;
}

.collector-digit-btn {
  min-width: 48px;
  padding: 10px 0;
  border-radius: 10px;
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 16px;
  font-weight: 800;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.collector-digit-btn.active {
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  color: white;
  box-shadow: var(--shadow-sm);
}

.collector-digit-btn:hover {
  transform: translateY(-1px);
}

.collector-confirm-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.collector-hint {
  font-size: 12px;
  color: var(--color-text-light);
}

.collector-btn {
  padding: 10px 14px;
  border-radius: 10px;
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 14px;
  font-weight: 700;
}

.collector-btn.primary {
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  color: white;
}

.collector-btn.secondary {
  background: rgba(108, 99, 255, 0.08);
  color: var(--color-primary);
}

.collector-btn.danger {
  background: rgba(255, 82, 82, 0.08);
  color: var(--color-danger, #FF5252);
}

.collector-btn.subtle {
  padding: 0;
  background: transparent;
  color: var(--color-primary);
  font-size: 13px;
}

.collector-export {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.collector-status {
  margin: 14px 0 0;
  font-size: 13px;
  color: var(--color-text-light);
}

.collector-status.saved {
  color: var(--color-success, #2ed573);
  font-weight: 700;
}

.result-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  background: var(--color-card);
  border-radius: 14px;
  box-shadow: var(--shadow-sm);
}

.placeholder-icon {
  font-size: 48px;
  color: var(--color-text-light);
  margin-bottom: 12px;
}

.result-placeholder p {
  font-size: 15px;
  color: var(--color-text-light);
}

.error-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  background: var(--color-card);
  border-radius: 14px;
  box-shadow: var(--shadow-sm);
  padding: 32px;
  text-align: center;
}

.error-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.error-display h3 {
  font-size: 18px;
  font-weight: 700;
  color: var(--color-danger);
  margin-bottom: 8px;
}

.error-message {
  font-size: 14px;
  color: var(--color-text-light);
  font-family: 'SF Mono', 'Fira Code', monospace;
  background: var(--color-bg);
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 12px;
  word-break: break-all;
}

.error-hint {
  font-size: 13px;
  color: var(--color-text-light);
}

.error-hint code {
  font-family: 'SF Mono', 'Fira Code', monospace;
  background: var(--color-bg);
  padding: 2px 8px;
  border-radius: 4px;
}

.step-progress {
  background: var(--color-card);
  border-radius: 14px;
  padding: 16px 20px;
  box-shadow: var(--shadow-sm);
}

.progress-bar {
  height: 6px;
  background: var(--color-bg);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 12px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--color-primary), var(--color-accent));
  border-radius: 3px;
  transition: width 0.3s ease;
}

.step-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.step-btn {
  padding: 8px 16px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 13px;
  background: var(--color-bg);
  color: var(--color-text);
}

.step-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.step-counter {
  font-size: 14px;
  font-weight: 700;
  color: var(--color-primary);
}

.step-display {
  background: var(--color-card);
  border-radius: 14px;
  padding: 24px;
  box-shadow: var(--shadow-sm);
  border-left: 4px solid var(--color-primary);
}

.step-header {
  margin-bottom: 12px;
}

.step-title {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 6px;
}

.step-code {
  display: inline-block;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  background: var(--color-bg);
  padding: 4px 12px;
  border-radius: 6px;
  color: var(--color-primary);
}

.step-desc {
  font-size: 15px;
  color: var(--color-text);
  margin-bottom: 10px;
}

.step-detail {
  font-size: 13px;
  line-height: 1.7;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 12px 16px;
  border-radius: 8px;
}

.step-pipeline {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  background: var(--color-card);
  border-radius: 14px;
  padding: 16px;
  box-shadow: var(--shadow-sm);
}

.pipeline-node {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 8px;
  cursor: pointer;
  transition: var(--transition);
  font-size: 12px;
  color: var(--color-text-light);
}

.pipeline-node:hover {
  background: var(--color-bg);
}

.pipeline-node.active {
  background: var(--color-primary);
  color: white;
}

.pipeline-node.done {
  color: var(--color-success);
}

.node-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}

/* Step visualization styles */
.step-visual {
  min-height: 200px;
}

.visual-block {
  background: var(--color-card);
  border-radius: 14px;
  padding: 20px;
  box-shadow: var(--shadow-sm);
}

.visual-title {
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 16px;
  color: var(--color-text);
}

.visual-subtitle {
  font-size: 13px;
  color: var(--color-text-light);
  margin-bottom: 14px;
}

.center-canvas {
  display: block;
  margin: 0 auto 12px;
  border-radius: 8px;
  border: 1px solid var(--color-border);
  width: 256px;
  height: 256px;
}

.visual-pixel-canvas {
  width: 196px;
  height: 196px;
  border-radius: 8px;
  image-rendering: pixelated;
  border: 1px solid var(--color-border);
  display: block;
  margin: 0 auto 12px;
}

.visual-stats {
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}

.stat-item {
  font-size: 13px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 4px 12px;
  border-radius: 8px;
}

/* Matrix multiplication flow diagram */
.matmul-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.matmul-flow.compact {
  gap: 10px;
}

.flow-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.flow-canvas {
  border-radius: 8px;
  border: 1px solid var(--color-border);
  image-rendering: pixelated;
}

.heatmap-canvas {
  background: #1a1a2e;
}

.flow-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-light);
  text-align: center;
  font-family: 'SF Mono', 'Fira Code', monospace;
  line-height: 1.4;
}

.flow-op {
  font-size: 24px;
  font-weight: 800;
  color: var(--color-primary);
  min-width: 24px;
  text-align: center;
}

.weight-placeholder {
  width: 100px;
  height: 100px;
  border-radius: 10px;
  background: linear-gradient(135deg, #2d2b55, #1a1a2e);
  border: 2px dashed var(--color-primary);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
}

.weight-placeholder.small {
  width: 80px;
  height: 80px;
}

.weight-count {
  font-size: 18px;
  font-weight: 800;
  color: var(--color-primary);
}

.weight-unit {
  font-size: 11px;
  color: var(--color-text-light);
}

.heatmap-legend-inline {
  display: flex;
  gap: 20px;
  justify-content: center;
  margin-top: 12px;
  font-size: 13px;
  color: var(--color-text-light);
}

.heatmap-legend-inline span {
  display: flex;
  align-items: center;
  gap: 6px;
}

.legend-dot {
  width: 14px;
  height: 14px;
  border-radius: 4px;
}

.legend-dot.positive {
  background: rgb(108, 99, 255);
}

.legend-dot.negative {
  background: rgb(255, 107, 157);
}

/* ReLU comparison */
.relu-comparison {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 16px;
}

.relu-side {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.relu-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-light);
}

.relu-arrow {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 700;
  color: var(--color-primary);
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.arrow-icon {
  font-size: 24px;
}

.relu-stats {
  display: flex;
  gap: 20px;
  justify-content: center;
}

.stat-active {
  font-size: 13px;
  color: rgb(108, 99, 255);
  background: rgba(108, 99, 255, 0.1);
  padding: 6px 14px;
  border-radius: 8px;
}

.stat-dead {
  font-size: 13px;
  color: var(--color-text-light);
  background: var(--color-bg);
  padding: 6px 14px;
  border-radius: 8px;
}

/* Output score bars (steps 4-5) */
.output-bars-mini {
  width: 140px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 8px;
  background: var(--color-bg);
  border-radius: 8px;
}

.out-bar-row {
  display: flex;
  align-items: center;
  gap: 4px;
  height: 14px;
}

.out-bar-digit {
  font-size: 10px;
  font-weight: 700;
  width: 14px;
  text-align: right;
  color: var(--color-text-light);
}

.out-bar-track {
  flex: 1;
  height: 8px;
  background: rgba(255,255,255,0.05);
  border-radius: 4px;
  overflow: hidden;
}

.out-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.out-bar-fill.positive {
  background: rgb(108, 99, 255);
}

.out-bar-fill.negative {
  background: rgb(255, 107, 157);
}

/* Score bars (step 5) */
.score-bars {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}

.score-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.score-digit {
  width: 24px;
  font-size: 16px;
  font-weight: 800;
  text-align: center;
  color: var(--color-text);
}

.score-track {
  flex: 1;
  height: 24px;
  background: var(--color-bg);
  border-radius: 12px;
  overflow: hidden;
}

.score-fill {
  height: 100%;
  border-radius: 12px;
  transition: width 0.5s ease;
}

.score-fill.positive {
  background: linear-gradient(90deg, rgb(108, 99, 255), rgb(108, 99, 255, 0.7));
}

.score-fill.negative {
  background: linear-gradient(90deg, rgb(255, 107, 157), rgb(255, 107, 157, 0.7));
}

.score-value {
  width: 60px;
  text-align: right;
  font-size: 13px;
  font-weight: 700;
  font-family: 'SF Mono', 'Fira Code', monospace;
  color: var(--color-text-light);
}

.score-value.positive {
  color: rgb(108, 99, 255);
}

.visual-hint {
  font-size: 13px;
  color: var(--color-text-light);
  text-align: center;
  margin-top: 4px;
}

/* Final prediction (step 7) */
.final-prediction {
  text-align: center;
}

.pred-hero {
  margin-bottom: 20px;
}

.pred-big-number {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100px;
  height: 100px;
  font-size: 56px;
  font-weight: 900;
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  color: white;
  border-radius: 24px;
  box-shadow: 0 8px 32px rgba(108, 99, 255, 0.3);
}

.pred-details {
  max-width: 500px;
  margin: 0 auto;
}

.confidence-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  padding: 12px 16px;
  background: var(--color-bg);
  border-radius: 12px;
}

.conf-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-light);
  white-space: nowrap;
}

.conf-bar-track {
  flex: 1;
  height: 12px;
  background: rgba(255,255,255,0.08);
  border-radius: 6px;
  overflow: hidden;
}

.conf-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--color-primary), var(--color-accent));
  border-radius: 6px;
  transition: width 0.5s ease;
}

.conf-value {
  font-size: 16px;
  font-weight: 800;
  color: var(--color-primary);
  white-space: nowrap;
}

/* Network diagram */
.network-block {
  background: var(--color-card);
  border-radius: 14px;
  padding: 20px;
  box-shadow: var(--shadow-sm);
}

.network-hint {
  font-size: 13px;
  color: var(--color-primary);
  font-weight: 600;
  margin-bottom: 12px;
}

.network-wrapper {
  border-radius: 10px;
  background: var(--color-bg);
  padding: 8px;
}

.code-reference {
  background: var(--color-card);
  border-radius: 14px;
  padding: 24px;
  box-shadow: var(--shadow-sm);
}

.code-reference h3 {
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 14px;
}

.code-block {
  background: #1e1e2e;
  border-radius: 10px;
  padding: 18px;
  overflow-x: auto;
}

.code-block pre {
  margin: 0;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 13px;
  line-height: 1.7;
  color: #cdd6f4;
}

.code-block .comment {
  color: #6c7086;
}

@media (max-width: 900px) {
  .main-layout {
    grid-template-columns: 1fr;
  }

  .collector-main {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
