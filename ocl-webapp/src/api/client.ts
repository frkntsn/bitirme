import {
  FeedbackItem,
  FeedbackResponse,
  HealthResponse,
  ResetMemoryResponse,
  SegmentResponse,
  StoredAnnotation,
} from '../types'
import { storedToApiPayload } from '../utils/annotationProjection'

const BASE = '/api'

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error(`Backend erişilemedi: ${res.status}`)
  return res.json()
}

export async function segment(
  imageBase64: string,
  annotations: StoredAnnotation[]
): Promise<SegmentResponse> {
  const res = await fetch(`${BASE}/segment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: imageBase64,
      annotations: storedToApiPayload(annotations),
    }),
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Backend hatası: ${res.status}`)
  }

  return res.json()
}

export async function infer(imageBase64: string): Promise<SegmentResponse> {
  const res = await fetch(`${BASE}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageBase64 }),
  })
  if (!res.ok) throw new Error(`Backend hatası: ${res.status}`)
  return res.json()
}

export async function sendFeedback(items: FeedbackItem[]): Promise<FeedbackResponse> {
  const res = await fetch(`${BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  })
  if (!res.ok) throw new Error(`Backend hatası: ${res.status}`)
  return res.json()
}

export async function resetMemory(): Promise<ResetMemoryResponse> {
  const res = await fetch(`${BASE}/reset-memory`, { method: 'POST' })
  if (!res.ok) throw new Error(`Backend hatası: ${res.status}`)
  return res.json()
}
