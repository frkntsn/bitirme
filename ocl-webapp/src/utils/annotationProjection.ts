import { Annotation, AnnotationInput, StoredAnnotation } from '../types'

const COLORS = [
  '#14b8a6', '#3b82f6', '#f43f5e', '#a855f7', '#f59e0b', '#06b6d4', '#84cc16',
]
let colorIdx = 0

export function nextAnnotationColor(): string {
  return COLORS[colorIdx++ % COLORS.length]
}

export function resetAnnotationColors(): void {
  colorIdx = 0
}

/** Çizim girişini orijinal piksel uzayında sakla */
export function inputToStored(input: AnnotationInput, color?: string): StoredAnnotation {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    shape: input.shape,
    label: input.label.trim() || 'etiket yok',
    color: color ?? nextAnnotationColor(),
    origX: input.origX,
    origY: input.origY,
    origRadius: input.origRadius,
    origBbox: input.origBbox,
    origPoints: input.origPoints,
  }
}

/** Orijinal koordinatları mevcut canvas ölçeğine yansıt */
export function projectAnnotations(
  stored: StoredAnnotation[],
  scaleX: number,
  scaleY: number
): Annotation[] {
  const sx = scaleX || 1
  const sy = scaleY || 1

  return stored.map((s) => {
    const base: Annotation = { ...s }

    if (s.shape === 'circle') {
      return {
        ...base,
        x: s.origX * sx,
        y: s.origY * sy,
        radius: s.origRadius * Math.min(sx, sy),
      }
    }

    if (s.shape === 'rect' && s.origBbox) {
      const [ox, oy, ow, oh] = s.origBbox
      return {
        ...base,
        rect: { x: ox * sx, y: oy * sy, width: ow * sx, height: oh * sy },
      }
    }

    if (s.shape === 'brush' && s.origPoints && s.origPoints.length >= 4) {
      const points: number[] = []
      for (let i = 0; i < s.origPoints.length; i += 2) {
        points.push(s.origPoints[i] * sx, s.origPoints[i + 1] * sy)
      }
      return { ...base, points }
    }

    return {
      ...base,
      x: s.origX * sx,
      y: s.origY * sy,
      radius: s.origRadius * Math.min(sx, sy),
    }
  })
}

export function storedToApiPayload(stored: StoredAnnotation[]) {
  return stored.map((a) => ({
    x: a.origX,
    y: a.origY,
    radius: a.origRadius,
    label: a.label,
    shape: a.shape,
    bbox: a.shape !== 'circle' && a.origBbox ? a.origBbox : undefined,
  }))
}
