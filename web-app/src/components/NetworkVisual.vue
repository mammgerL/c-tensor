<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  step: { type: Number, default: 7 },
  hiddenActivations: { type: Array, default: () => [] },
  outputActivations: { type: Array, default: () => [] },
  preRelu: { type: Array, default: () => [] },
  preSoftmax: { type: Array, default: () => [] },
  predicted: { type: Number, default: -1 },
})

const hoveredNode = ref(null)

const svgWidth = 700
const svgHeight = 480
const padding = { top: 44, bottom: 24, left: 60, right: 60 }

const layerDefs = [
  { name: 'input', label: '输入层 (784)', displaySize: 16, fullSize: 784 },
  { name: 'hidden', label: '隐藏层 (256)', displaySize: 24, fullSize: 256 },
  { name: 'output', label: '输出层 (10)', displaySize: 10, fullSize: 10 },
]

const layerX = [padding.left + 30, svgWidth / 2, svgWidth - padding.right - 30]

// Step mapping:
// 0: input        → input layer lit
// 1: matmul1      → input→hidden connections lit
// 2: bias1        → hidden nodes lit (pre_relu values)
// 3: relu         → hidden nodes update (post-relu, some go dark)
// 4: matmul2      → hidden→output connections lit
// 5: bias2        → output nodes lit (pre_softmax values)
// 6: logsoftmax   → output nodes update (probabilities)
// 7: argmax       → winner highlighted

function layerVisible(layerName) {
  if (layerName === 'input') return true
  if (layerName === 'hidden') return props.step >= 2
  if (layerName === 'output') return props.step >= 5
  return false
}

function layerActive(layerName) {
  if (layerName === 'input') return props.step === 0
  if (layerName === 'hidden') return props.step >= 2 && props.step <= 3
  if (layerName === 'output') return props.step >= 5
  return false
}

function linksVisible(fromLayer) {
  if (fromLayer === 'input') return props.step >= 1
  if (fromLayer === 'hidden') return props.step >= 4
  return false
}

function linksActive(fromLayer) {
  if (fromLayer === 'input') return props.step === 1
  if (fromLayer === 'hidden') return props.step === 4
  return false
}

// Get activation value for a node depending on current step
function getActivation(layerName, displayIndex) {
  if (layerName === 'hidden') {
    const data = props.step >= 3 ? props.hiddenActivations : props.preRelu
    if (!data.length) return 0
    const step = Math.floor(data.length / 24)
    return data[displayIndex * step] || 0
  }
  if (layerName === 'output') {
    if (props.step >= 6 && props.outputActivations.length) {
      return Math.exp(props.outputActivations[displayIndex]) || 0
    }
    if (props.step >= 5 && props.preSoftmax.length) {
      // Normalize pre_softmax to [0,1] for display
      const max = Math.max(...props.preSoftmax.map(Math.abs), 0.001)
      return Math.max(0, props.preSoftmax[displayIndex] / max)
    }
    return 0
  }
  return 0
}

const layers = computed(() => {
  return layerDefs.map((def, li) => {
    const x = layerX[li]
    const availH = svgHeight - padding.top - padding.bottom
    const spacing = availH / (def.displaySize + 1)
    const nodes = []
    for (let i = 0; i < def.displaySize; i++) {
      nodes.push({
        layer: def.name,
        index: i,
        realIndex: def.name === 'hidden' ? i * Math.floor(256 / def.displaySize) : i,
        x,
        y: padding.top + spacing * (i + 1),
        activation: getActivation(def.name, i),
      })
    }
    return { ...def, x, nodes }
  })
})

const links = computed(() => {
  const result = []
  for (let li = 0; li < layers.value.length - 1; li++) {
    const srcLayer = layers.value[li]
    const tgtLayer = layers.value[li + 1]
    const visible = linksVisible(srcLayer.name)
    const active = linksActive(srcLayer.name)
    for (const src of srcLayer.nodes) {
      for (const tgt of tgtLayer.nodes) {
        const strength = visible ? Math.abs(tgt.activation) : 0
        result.push({
          x1: src.x, y1: src.y,
          x2: tgt.x, y2: tgt.y,
          strength,
          fromLayer: srcLayer.name,
          visible,
          active,
        })
      }
    }
  }
  return result
})

function linkStroke(link) {
  if (!link.visible) return '#333'
  if (link.active) return link.strength > 0.05 ? '#6C63FF' : '#556'
  return link.strength > 0.05 ? '#6677aa' : '#444'
}

function linkOpacity(link) {
  if (!link.visible) return 0.03
  if (link.active) return 0.08 + Math.min(0.5, link.strength * 0.6)
  return 0.03 + Math.min(0.2, link.strength * 0.15)
}

function linkWidth(link) {
  if (!link.visible) return 0.3
  if (link.active && link.strength > 0.1) return 1.5
  return 0.5
}

function nodeColor(node) {
  const vis = layerVisible(node.layer)
  if (!vis) return '#2a2a3a'

  if (node.layer === 'input') return props.step === 0 ? '#5588cc' : '#4a6080'

  if (node.activation <= 0.001) return '#2a2a3a' // dead

  const maxVal = maxActivation(node.layer)
  const t = Math.min(1, Math.abs(node.activation) / maxVal)

  if (node.layer === 'output') {
    if (props.step >= 7 && node.index === props.predicted) {
      return `rgb(${Math.round(40 + 215 * t)}, ${Math.round(180 * t)}, ${Math.round(60 + 80 * t)})`
    }
    return `rgba(108, 99, 255, ${0.3 + t * 0.7})`
  }

  // hidden: cyan gradient
  return `rgb(${Math.round(20 + 180 * t)}, ${Math.round(20 + 220 * t)}, ${Math.round(60 + 195 * t)})`
}

function nodeRadius(node) {
  if (node.layer === 'output') return 16
  if (node.layer === 'hidden') return 6
  return 5
}

function nodeStroke(node) {
  if (hoveredNode.value === node) return '#fff'
  if (layerActive(node.layer)) return 'rgba(108, 99, 255, 0.6)'
  if (props.step >= 7 && node.layer === 'output' && node.index === props.predicted) return '#4CAF50'
  return 'transparent'
}

function nodeStrokeWidth(node) {
  if (hoveredNode.value === node) return 2
  if (layerActive(node.layer)) return 1.5
  if (props.step >= 7 && node.layer === 'output' && node.index === props.predicted) return 2.5
  return 0
}

function nodeOpacity(node) {
  return layerVisible(node.layer) ? 1 : 0.15
}

function labelOpacity(li) {
  const name = layerDefs[li].name
  if (name === 'input') return 1
  if (name === 'hidden') return props.step >= 1 ? 1 : 0.25
  if (name === 'output') return props.step >= 4 ? 1 : 0.25
  return 0.25
}

function maxActivation(layerName) {
  if (layerName === 'hidden') {
    const data = props.step >= 3 ? props.hiddenActivations : props.preRelu
    return data.length ? Math.max(...data, 0.001) : 1
  }
  if (layerName === 'output') {
    if (props.step >= 6 && props.outputActivations.length) {
      return Math.max(...props.outputActivations.map(v => Math.exp(v)), 0.001)
    }
    if (props.step >= 5 && props.preSoftmax.length) {
      return Math.max(...props.preSoftmax.map(Math.abs), 0.001)
    }
  }
  return 1
}

// Step annotation markers
const stepAnnotations = computed(() => {
  const s = props.step
  const annotations = []
  const midInputHidden = (layerX[0] + layerX[1]) / 2
  const midHiddenOutput = (layerX[1] + layerX[2]) / 2

  if (s === 0) {
    annotations.push({ x: layerX[0], y: svgHeight - 6, text: '← 当前步骤', color: '#6C63FF' })
  } else if (s === 1) {
    annotations.push({ x: midInputHidden, y: svgHeight / 2, text: '× W1', color: '#6C63FF' })
  } else if (s === 2) {
    annotations.push({ x: layerX[1], y: svgHeight - 6, text: '+ b1', color: '#6C63FF' })
  } else if (s === 3) {
    annotations.push({ x: layerX[1], y: svgHeight - 6, text: 'ReLU', color: '#FF6B9D' })
  } else if (s === 4) {
    annotations.push({ x: midHiddenOutput, y: svgHeight / 2, text: '× W2', color: '#6C63FF' })
  } else if (s === 5) {
    annotations.push({ x: layerX[2], y: svgHeight - 6, text: '+ b2', color: '#6C63FF' })
  } else if (s === 6) {
    annotations.push({ x: layerX[2], y: svgHeight - 6, text: 'softmax', color: '#6C63FF' })
  } else if (s === 7) {
    annotations.push({ x: layerX[2], y: svgHeight - 6, text: 'argmax', color: '#4CAF50' })
  }
  return annotations
})
</script>

<template>
  <div class="network-visual">
    <svg class="network-svg" :viewBox="`0 0 ${svgWidth} ${svgHeight}`" preserveAspectRatio="xMidYMid meet">
      <!-- Connections -->
      <line
        v-for="(link, i) in links" :key="'l' + i"
        :x1="link.x1" :y1="link.y1" :x2="link.x2" :y2="link.y2"
        :stroke="linkStroke(link)"
        :stroke-opacity="linkOpacity(link)"
        :stroke-width="linkWidth(link)"
      />

      <!-- Nodes -->
      <template v-for="(layer, li) in layers" :key="li">
        <g
          v-for="node in layer.nodes" :key="node.layer + node.index"
          class="node-group"
          :opacity="nodeOpacity(node)"
          @mouseenter="hoveredNode = node"
          @mouseleave="hoveredNode = null"
        >
          <circle
            :cx="node.x" :cy="node.y"
            :r="nodeRadius(node)"
            :fill="nodeColor(node)"
            :stroke="nodeStroke(node)"
            :stroke-width="nodeStrokeWidth(node)"
          />
          <text
            v-if="layer.name === 'output'"
            :x="node.x + 22" :y="node.y + 5"
            class="output-label"
            :class="{ predicted: step >= 7 && node.index === predicted }"
            :opacity="nodeOpacity(node)"
          >{{ node.index }}</text>
        </g>
      </template>

      <!-- Layer labels -->
      <text
        v-for="(layer, li) in layers" :key="'lbl' + li"
        :x="layer.x" :y="16"
        text-anchor="middle"
        class="layer-label"
        :opacity="labelOpacity(li)"
      >{{ layer.label }}</text>

      <!-- Sample count -->
      <text
        v-for="(layer, li) in layers" :key="'cnt' + li"
        v-show="layer.displaySize < layer.fullSize"
        :x="layer.x" :y="28"
        text-anchor="middle"
        class="sample-label"
        :opacity="labelOpacity(li) * 0.6"
      >({{ layer.displaySize }}/{{ layer.fullSize }} 采样)</text>

      <!-- Step annotation -->
      <g v-for="(ann, i) in stepAnnotations" :key="'ann' + i">
        <text
          :x="ann.x" :y="ann.y"
          text-anchor="middle"
          class="step-annotation"
          :fill="ann.color"
        >{{ ann.text }}</text>
      </g>
    </svg>

    <!-- Hover tooltip -->
    <div v-if="hoveredNode" class="tooltip">
      <span class="tooltip-layer">{{ hoveredNode.layer === 'input' ? '输入' : hoveredNode.layer === 'hidden' ? '隐藏层' : '输出' }} #{{ hoveredNode.realIndex }}</span>
      <span class="tooltip-value" :class="{ active: hoveredNode.activation > 0 }">
        {{ hoveredNode.layer === 'input' ? '采样节点' : hoveredNode.activation > 0.001 ? `激活值: ${hoveredNode.activation.toFixed(4)}` : '沉默 (0)' }}
      </span>
      <span v-if="hoveredNode.layer === 'output' && step >= 6" class="tooltip-prob">
        概率: {{ (hoveredNode.activation * 100).toFixed(1) }}%
      </span>
    </div>
  </div>
</template>

<style scoped>
.network-visual {
  position: relative;
  width: 100%;
}

.network-svg {
  width: 100%;
  height: auto;
  display: block;
}

.node-group {
  cursor: pointer;
  transition: opacity 0.3s ease;
}

.node-group circle {
  transition: fill 0.4s ease, stroke 0.3s ease, r 0.2s ease;
}

.node-group:hover circle {
  r: 10;
}

.layer-label {
  font-size: 13px;
  font-weight: 700;
  fill: var(--color-text-light, #999);
  transition: opacity 0.3s ease;
}

.sample-label {
  font-size: 10px;
  fill: var(--color-text-light, #777);
  transition: opacity 0.3s ease;
}

.output-label {
  font-size: 14px;
  font-weight: 700;
  fill: var(--color-text-light, #999);
  transition: opacity 0.3s ease;
}

.output-label.predicted {
  fill: #4CAF50;
  font-size: 16px;
  font-weight: 800;
}

.step-annotation {
  font-size: 15px;
  font-weight: 800;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.tooltip {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(30, 30, 40, 0.95);
  color: white;
  padding: 12px 16px;
  border-radius: 12px;
  font-size: 13px;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  gap: 4px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

.tooltip-layer {
  font-weight: 700;
}

.tooltip-value {
  color: #888;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

.tooltip-value.active {
  color: rgb(100, 220, 240);
}

.tooltip-prob {
  color: #6C63FF;
  font-weight: 600;
}
</style>
