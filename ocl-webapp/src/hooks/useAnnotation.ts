import { useState, useCallback } from 'react'
import { Annotation, AnnotationInput } from '../types'

const COLORS = [
  '#2563eb', '#16a34a', '#dc2626', '#9333ea',
  '#db2777', '#d97706', '#0891b2', '#65a30d',
]

let colorIdx = 0

function nextColor(): string {
  return COLORS[colorIdx++ % COLORS.length]
}

export function useAnnotation() {
  const [annotations, setAnnotations] = useState<Annotation[]>([])

  const addAnnotation = useCallback((input: AnnotationInput) => {
    const annotation: Annotation = {
      id: `${Date.now()}-${Math.random()}`,
      shape: input.shape,
      label: input.label.trim() || 'etiket yok',
      color: nextColor(),
      origX: input.origX,
      origY: input.origY,
      origRadius: input.origRadius,
      origBbox: input.origBbox,
      x: input.x,
      y: input.y,
      radius: input.radius,
      rect: input.rect,
      points: input.points,
    }
    setAnnotations((prev) => [...prev, annotation])
    return annotation
  }, [])

  const removeAnnotation = useCallback((id: string) => {
    setAnnotations((prev) => prev.filter((a) => a.id !== id))
  }, [])

  const clearAll = useCallback(() => {
    setAnnotations([])
    colorIdx = 0
  }, [])

  const replaceAll = useCallback((items: Annotation[]) => {
    setAnnotations(items)
    colorIdx = items.length
  }, [])

  return {
    annotations,
    setAnnotations,
    addAnnotation,
    removeAnnotation,
    clearAll,
    replaceAll,
  }
}
