const API_BASE = '/api'

export async function fetchArchitecture() {
  const res = await fetch(`${API_BASE}/architecture`)
  if (!res.ok) throw new Error('Failed to fetch architecture')
  return res.json()
}

export async function fetchPredict(index) {
  const res = await fetch(`${API_BASE}/predict?index=${index}`)
  if (!res.ok) throw new Error('Failed to fetch prediction')
  return res.json()
}

export async function fetchEval() {
  const res = await fetch(`${API_BASE}/eval`)
  if (!res.ok) throw new Error('Failed to fetch eval stats')
  return res.json()
}

export async function fetchPredictWithPixels(pixelArray) {
  const res = await fetch(`${API_BASE}/predict_pixels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pixels: pixelArray }),
  })
  if (!res.ok) throw new Error('Failed to predict custom pixels')
  return res.json()
}