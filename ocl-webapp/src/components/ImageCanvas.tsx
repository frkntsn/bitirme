import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Stage, Layer, Image as KonvaImage, Circle, Rect, Text } from 'react-konva'
import { Annotation, DetectedRegion } from '../types'

interface Props {
  imageUrl: string | null
  annotations: Annotation[]
  detections: DetectedRegion[]
  onAddAnnotation: (x: number, y: number, radius: number) => void
  containerSize: { width: number; height: number }
  onScaleChange: (scaleX: number, scaleY: number) => void
}

// Etiket bazlı renk üret
const LABEL_COLORS: Record<string, string> = {}
const PALETTE = ['#2563eb','#16a34a','#dc2626','#9333ea','#db2777','#d97706','#0891b2']
let paletteIdx = 0
function colorForLabel(label: string): string {
  if (!LABEL_COLORS[label]) {
    LABEL_COLORS[label] = PALETTE[paletteIdx++ % PALETTE.length]
  }
  return LABEL_COLORS[label]
}

export default function ImageCanvas({
  imageUrl, annotations, detections,
  onAddAnnotation, containerSize, onScaleChange,
}: Props) {
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [stageSize, setStageSize] = useState({ width: 600, height: 450 })
  const [scale, setScale] = useState({ x: 1, y: 1 })
  const [drawing, setDrawing] = useState(false)
  const [preview, setPreview] = useState<{ x: number; y: number; r: number } | null>(null)
  const startPos = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    if (!imageUrl) return
    const img = new window.Image()
    img.onload = () => {
      const maxW = containerSize.width - 2
      const maxH = containerSize.height - 2
      const s = Math.min(maxW / img.width, maxH / img.height, 1)
      setStageSize({ width: Math.round(img.width * s), height: Math.round(img.height * s) })
      setScale({ x: s, y: s })
      onScaleChange(s, s)
      setImage(img)
    }
    img.src = imageUrl
  }, [imageUrl, containerSize, onScaleChange])

  const handleMouseDown = useCallback((e: any) => {
    if (!image) return
    const pos = e.target.getStage().getPointerPosition()
    if (!pos) return
    startPos.current = pos
    setDrawing(true)
  }, [image])

  const handleMouseMove = useCallback((e: any) => {
    if (!drawing || !startPos.current) return
    const pos = e.target.getStage().getPointerPosition()
    if (!pos) return
    setPreview({ x: startPos.current.x, y: startPos.current.y, r: Math.hypot(pos.x - startPos.current.x, pos.y - startPos.current.y) })
  }, [drawing])

  const handleMouseUp = useCallback((e: any) => {
    if (!drawing || !startPos.current) return
    setDrawing(false)
    const pos = e.target.getStage().getPointerPosition()
    if (!pos) return
    const r = Math.hypot(pos.x - startPos.current.x, pos.y - startPos.current.y)
    if (r > 8) onAddAnnotation(startPos.current.x, startPos.current.y, r)
    setPreview(null)
    startPos.current = null
  }, [drawing, onAddAnnotation])

  if (!imageUrl) return null

  return (
    <Stage
      width={stageSize.width}
      height={stageSize.height}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      style={{ cursor: 'crosshair', display: 'block' }}
    >
      <Layer>
        {image && <KonvaImage image={image} width={stageSize.width} height={stageSize.height} />}

        {/* Tespit edilen bölgeler — bbox dikdörtgenleri */}
        {detections.map((d, i) => {
          const [ox, oy, ow, oh] = d.bbox
          const x = ox * scale.x
          const y = oy * scale.y
          const w = ow * scale.x
          const h = oh * scale.y
          const color = colorForLabel(d.label)
          const pct = Math.round(d.confidence * 100)
          return (
            <React.Fragment key={`det-${i}`}>
              <Rect x={x} y={y} width={w} height={h} stroke={color} strokeWidth={2} fill="transparent" />
              {/* Etiket etiketi */}
              <Rect x={x} y={y - 18} width={Math.max(w, 70)} height={16} fill={color} cornerRadius={3} />
              <Text
                x={x + 4} y={y - 16}
                text={`${d.label} ${pct}%`}
                fontSize={11} fill="white"
              />
            </React.Fragment>
          )
        })}

        {/* Kullanıcı annotasyonları — daireler */}
        {annotations.map((ann) => (
          <React.Fragment key={ann.id}>
            <Circle x={ann.x} y={ann.y} radius={ann.radius} stroke={ann.color} strokeWidth={2.5} fill="transparent" />
            <Rect x={ann.x - 40} y={ann.y - ann.radius - 22} width={80} height={18} fill={ann.color} cornerRadius={4} />
            <Text x={ann.x - 40} y={ann.y - ann.radius - 22} width={80} height={18} text={ann.label} fontSize={11} fill="white" align="center" verticalAlign="middle" />
          </React.Fragment>
        ))}

        {/* Çizim önizleme */}
        {preview && (
          <Circle x={preview.x} y={preview.y} radius={preview.r} stroke="#888" strokeWidth={1.5} dash={[5, 3]} fill="transparent" />
        )}
      </Layer>
    </Stage>
  )
}
