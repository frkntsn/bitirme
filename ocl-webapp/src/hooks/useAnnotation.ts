import { useState, useCallback } from 'react'
import { Annotation } from '../types'

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

  const addAnnotation = useCallback(
    (
      x: number, y: number, radius: number, label: string,
      scaleX: number, scaleY: number
    ) => {
      const annotation: Annotation = {
        id: `${Date.now()}-${Math.random()}`,
        x,
        y,
        radius,
        label: label.trim() || 'etiket yok',
        color: nextColor(),
        origX: Math.round(x / scaleX),
        origY: Math.round(y / scaleY),
        origRadius: Math.round(radius / Math.min(scaleX, scaleY)),
      }
      setAnnotations((prev) => [...prev, annotation])
      return annotation
    },
    []
  )

  const removeAnnotation = useCallback((id: string) => {
    setAnnotations((prev) => prev.filter((a) => a.id !== id))
  }, [])

  const clearAll = useCallback(() => {
    setAnnotations([])
    colorIdx = 0
  }, [])

  return { annotations, addAnnotation, removeAnnotation, clearAll }
}
