/**
 * Pure JS MNIST inference engine.
 * Loads weights from binary and runs forward pass: 784 -> 256 (ReLU) -> 10 (logsoftmax)
 * No dependencies, no WASM, no server needed.
 */

export class MnistInference {
  constructor() {
    this.w1 = null  // Float32Array, shape [784, 256]
    this.b1 = null  // Float32Array, shape [256]
    this.w2 = null  // Float32Array, shape [256, 10]
    this.b2 = null  // Float32Array, shape [10]
    this.loaded = false
  }

  /**
   * Load weights from a binary file (same format as mnist_mlp.bin).
   * Header: 32 bytes, then raw floats.
   */
  async loadWeights(url) {
    const resp = await fetch(url)
    if (!resp.ok) throw new Error(`Failed to load weights: ${resp.status}`)
    const buf = await resp.arrayBuffer()
    this._parseWeights(new DataView(buf))
  }

  /**
   * Load weights from an ArrayBuffer (e.g. imported as a module).
   */
  loadWeightsFromBuffer(buf) {
    this._parseWeights(new DataView(buf))
  }

  _parseWeights(dv) {
    // Header: magic(u32), version(u32), w1_rows(i32), w1_cols(i32), b1_len(i32), w2_rows(i32), w2_cols(i32), b2_len(i32)
    const magic = dv.getUint32(0, true)
    const version = dv.getUint32(4, true)
    if (magic !== 0x314C504D) throw new Error(`Invalid model magic: 0x${magic.toString(16)}`)
    if (version !== 1) throw new Error(`Unsupported model version: ${version}`)

    const w1Rows = dv.getInt32(8, true)
    const w1Cols = dv.getInt32(12, true)
    const b1Len = dv.getInt32(16, true)
    const w2Rows = dv.getInt32(20, true)
    const w2Cols = dv.getInt32(24, true)
    const b2Len = dv.getInt32(28, true)

    let offset = 32  // after header

    const readFloatArray = (count) => {
      const bytes = count * 4
      const arr = new Float32Array(count)
      for (let i = 0; i < count; i++) {
        arr[i] = dv.getFloat32(offset + i * 4, true)
      }
      offset += bytes
      return arr
    }

    this.w1 = readFloatArray(w1Rows * w1Cols)  // [784, 256]
    this.b1 = readFloatArray(b1Len)             // [256]
    this.w2 = readFloatArray(w2Rows * w2Cols)  // [256, 10]
    this.b2 = readFloatArray(b2Len)             // [10]
    this.loaded = true
  }

  /**
   * Forward pass on a single 784-dim input.
   * Returns detailed trace for visualization.
   */
  predict(pixels) {
    if (!this.loaded) throw new Error('Weights not loaded')

    const H = this.b1.length  // 256
    const pixels32 = new Float32Array(pixels)

    // Step 1: matmul1 = pixels @ W1  -> [1, 256]
    // C[0,j] = sum_i pixels[i] * W1[i*H + j]
    const matmul1 = new Float32Array(H)
    for (let j = 0; j < H; j++) {
      let sum = 0
      for (let i = 0; i < 784; i++) {
        sum += pixels32[i] * this.w1[i * H + j]
      }
      matmul1[j] = sum
    }

    // Step 2: pre_relu = matmul1 + b1
    const preRelu = new Float32Array(H)
    for (let j = 0; j < H; j++) {
      preRelu[j] = matmul1[j] + this.b1[j]
    }

    // Step 3: hidden = ReLU(pre_relu)
    const hidden = new Float32Array(H)
    for (let j = 0; j < H; j++) {
      hidden[j] = preRelu[j] > 0 ? preRelu[j] : 0
    }

    // Step 4: matmul2 = hidden @ W2  -> [1, 10]
    // C[0,c] = sum_h hidden[h] * W2[h*10 + c]
    const matmul2 = new Float32Array(10)
    for (let c = 0; c < 10; c++) {
      let sum = 0
      for (let h = 0; h < H; h++) {
        sum += hidden[h] * this.w2[h * 10 + c]
      }
      matmul2[c] = sum
    }

    // Step 5: pre_softmax = matmul2 + b2
    const preSoftmax = new Float32Array(10)
    for (let c = 0; c < 10; c++) {
      preSoftmax[c] = matmul2[c] + this.b2[c]
    }

    // Step 6: log_softmax
    const maxVal = Math.max(...preSoftmax)
    let sumExp = 0
    const centered = new Float32Array(10)
    for (let c = 0; c < 10; c++) {
      centered[c] = preSoftmax[c] - maxVal
      sumExp += Math.exp(centered[c])
    }
    const logSumExp = Math.log(sumExp)

    const output = new Float32Array(10)
    for (let c = 0; c < 10; c++) {
      output[c] = preSoftmax[c] - maxVal - logSumExp
    }

    // Step 7: argmax
    let predicted = 0
    let bestVal = output[0]
    for (let c = 1; c < 10; c++) {
      if (output[c] > bestVal) {
        bestVal = output[c]
        predicted = c
      }
    }
    const confidence = Math.exp(bestVal)

    return {
      pixels: Array.from(pixels32),
      predicted,
      confidence,
      hidden: Array.from(hidden),
      pre_relu: Array.from(preRelu),
      matmul1: Array.from(matmul1),
      output: Array.from(output),
      pre_softmax: Array.from(preSoftmax),
      matmul2: Array.from(matmul2),
    }
  }

  /**
   * Lightweight forward pass: returns only {predicted, confidence}.
   * Avoids allocating intermediate arrays for traces — use when scoring many
   * samples where you don't need the full activation trace per sample.
   */
  predictFast(pixels) {
    if (!this.loaded) throw new Error('Weights not loaded')

    const H = this.b1.length
    const w1 = this.w1, b1 = this.b1, w2 = this.w2, b2 = this.b2

    // hidden = ReLU(x @ W1 + b1)
    const hidden = new Float32Array(H)
    for (let j = 0; j < H; j++) {
      let sum = b1[j]
      for (let i = 0; i < 784; i++) {
        sum += pixels[i] * w1[i * H + j]
      }
      hidden[j] = sum > 0 ? sum : 0
    }

    // logits = hidden @ W2 + b2
    const logits = new Float32Array(10)
    let maxLogit = -Infinity
    for (let c = 0; c < 10; c++) {
      let sum = b2[c]
      for (let h = 0; h < H; h++) {
        sum += hidden[h] * w2[h * 10 + c]
      }
      logits[c] = sum
      if (sum > maxLogit) maxLogit = sum
    }

    // softmax, argmax
    let sumExp = 0
    for (let c = 0; c < 10; c++) sumExp += Math.exp(logits[c] - maxLogit)
    let predicted = 0
    for (let c = 1; c < 10; c++) if (logits[c] > logits[predicted]) predicted = c
    const confidence = Math.exp(logits[predicted] - maxLogit) / sumExp

    return { predicted, confidence }
  }
}

/**
 * Load MNIST test-samples binary file (generated by generate_test_samples.py).
 * Returns { count, dim, pixelsU8: Uint8Array (count * 784), labels: Uint8Array (count) }.
 */
export async function loadTestSamples(url) {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`Failed to load samples: ${resp.status}`)
  const buf = await resp.arrayBuffer()
  const dv = new DataView(buf)

  const magic = dv.getUint32(0, true)
  if (magic !== 0x54534E4D) throw new Error(`Invalid samples magic: 0x${magic.toString(16)}`)
  const count = dv.getUint32(4, true)
  const dim = dv.getUint32(8, true)
  const numClasses = dv.getUint32(12, true)
  if (dim !== 784 || numClasses !== 10) {
    throw new Error(`Unexpected sample shape: dim=${dim}, classes=${numClasses}`)
  }

  const stride = dim + 1  // 784 pixels + 1 label byte
  const pixelsU8 = new Uint8Array(count * dim)
  const labels = new Uint8Array(count)
  const body = new Uint8Array(buf, 16)
  for (let i = 0; i < count; i++) {
    const base = i * stride
    pixelsU8.set(body.subarray(base, base + dim), i * dim)
    labels[i] = body[base + dim]
  }
  return { count, dim, pixelsU8, labels }
}

/**
 * Normalize a uint8 pixel row [0-255] into a Float32Array in [-1, 1].
 */
export function normalizePixels(pixelsU8, offset = 0, length = 784) {
  const out = new Float32Array(length)
  for (let i = 0; i < length; i++) {
    out[i] = pixelsU8[offset + i] / 127.5 - 1.0
  }
  return out
}
