import { Annotation, SegmentResponse } from '../types'

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
