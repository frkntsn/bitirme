import { Annotation, FeedbackItem, FeedbackResponse, SegmentResponse } from '../types'

const BASE = '/api'

export async function segment(
  imageBase64: string,
  annotations: Annotation[]
): Promise<SegmentResponse> {
  const payload = {
    image: imageBase64,
    annotations: annotations.map((a) => ({
      x: a.origX,
      y: a.origY,
      radius: a.origRadius,
      label: a.label,
    })),
  }

  const res = await fetch(`${BASE}/segment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    throw new Error(`Backend hatası: ${res.status}`)
  }

  return res.json()
}

export async function infer(imageBase64: string): Promise<SegmentResponse> {
  const res = await fetch(`${BASE}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageBase64 }),
  })
  if (!res.ok) {
    throw new Error(`Backend hatası: ${res.status}`)
  }
  return res.json()
}

export async function sendFeedback(items: FeedbackItem[]): Promise<FeedbackResponse> {
  const res = await fetch(`${BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  })
  if (!res.ok) {
    throw new Error(`Backend hatası: ${res.status}`)
  }
  return res.json()
}
