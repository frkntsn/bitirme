import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Stage, Layer, Group, Image as KonvaImage, Circle, Rect, Text, Line } from 'react-konva'
import type Konva from 'konva'
import { Annotation, AnnotationInput, AnnotationTool, DetectedRegion } from '../types'
import { bboxFromCircle, bboxFromPoints, bboxFromRect } from '../utils/annotationGeometry'
import styles from './ImageCanvas.module.css'

interface Props {
  imageUrl: string | null
  annotations: Annotation[]
  detections: DetectedRegion[]
  tool: AnnotationTool
  brushSize: number
  label: string
  selectedDetectionIdx: number | null
  onSelectDetection: (idx: number | null) => void
  onAddAnnotation: (input: AnnotationInput) => void
  containerSize: { width: number; height: number }
  onScaleChange: (scaleX: number, scaleY: number) => void
}

const LABEL_COLORS: Record<string, string> = {}
const PALETTE = ['#0d9488', '#2563eb', '#dc2626', '#9333ea', '#db2777', '#d97706', '#0891b2']
let paletteIdx = 0

function colorForLabel(label: string): string {
  if (!LABEL_COLORS[label]) {
    LABEL_COLORS[label] = PALETTE[paletteIdx++ % PALETTE.length]
  }
  return LABEL_COLORS[label]
}

const MIN_STROKE = 12
const MIN_RADIUS = 8
const ZOOM_MIN = 0.35
const ZOOM_MAX = 5

function brushBounds(points: number[]) {
  const xs = points.filter((_, i) => i % 2 === 0)
  const ys = points.filter((_, i) => i % 2 === 1)
  return {
    x: Math.min(...xs),
    y: Math.min(...ys),
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys),
  }
}

export default function ImageCanvas({
  imageUrl,
  annotations,
  detections,
  tool,
  brushSize,
  label,
  selectedDetectionIdx,
  onSelectDetection,
  onAddAnnotation,
  containerSize,
  onScaleChange,
}: Props) {
  const groupRef = useRef<Konva.Group>(null)
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [stageSize, setStageSize] = useState({ width: 600, height: 450 })
  const [fitScale, setFitScale] = useState({ x: 1, y: 1 })
  const [zoom, setZoom] = useState(1)
  const [groupPos, setGroupPos] = useState({ x: 0, y: 0 })
  const [drawing, setDrawing] = useState(false)
  const [previewCircle, setPreviewCircle] = useState<{ x: number; y: number; r: number } | null>(null)
  const [previewRect, setPreviewRect] = useState<{ x: number; y: number; width: number; height: number } | null>(null)
  const [brushStroke, setBrushStroke] = useState<number[]>([])
  const startPos = useRef<{ x: number; y: number } | null>(null)

  const centerContent = useCallback(
    (z: number, imgW: number, imgH: number) => {
      const x = (containerSize.width - imgW * z) / 2
      const y = (containerSize.height - imgH * z) / 2
      return { x, y }
    },
    [containerSize]
  )

  useEffect(() => {
    if (!imageUrl) return
    const img = new window.Image()
    img.onload = () => {
      const maxW = containerSize.width - 2
      const maxH = containerSize.height - 2
      const s = Math.min(maxW / img.width, maxH / img.height, 1)
      const w = Math.round(img.width * s)
      const h = Math.round(img.height * s)
      setStageSize({ width: w, height: h })
      setFitScale({ x: s, y: s })
      onScaleChange(s, s)
      setImage(img)
      setZoom(1)
      setGroupPos(centerContent(1, w, h))
    }
    img.src = imageUrl
  }, [imageUrl, containerSize, onScaleChange, centerContent])

  const resetView = useCallback(() => {
    setZoom(1)
    setGroupPos(centerContent(1, stageSize.width, stageSize.height))
    const g = groupRef.current
    if (g) {
      g.scale({ x: 1, y: 1 })
      g.position(centerContent(1, stageSize.width, stageSize.height))
    }
  }, [centerContent, stageSize])

  const applyZoom = useCallback(
    (direction: 1 | -1, pointer?: { x: number; y: number }) => {
      const g = groupRef.current
      if (!g) return
      const scaleBy = 1.12
      const oldScale = g.scaleX()
      const ptr = pointer ?? {
        x: containerSize.width / 2,
        y: containerSize.height / 2,
      }
      const mousePointTo = {
        x: (ptr.x - g.x()) / oldScale,
        y: (ptr.y - g.y()) / oldScale,
      }
      const newScale =
        direction > 0
          ? Math.min(ZOOM_MAX, oldScale * scaleBy)
          : Math.max(ZOOM_MIN, oldScale / scaleBy)
      g.scale({ x: newScale, y: newScale })
      const pos = {
        x: ptr.x - mousePointTo.x * newScale,
        y: ptr.y - mousePointTo.y * newScale,
      }
      g.position(pos)
      setZoom(newScale)
      setGroupPos(pos)
    },
    [containerSize]
  )

  const handleWheel = useCallback(
    (e: Konva.KonvaEventObject<WheelEvent>) => {
      e.evt.preventDefault()
      const stage = e.target.getStage()
      const pointer = stage?.getPointerPosition()
      if (!pointer) return
      applyZoom(e.evt.deltaY < 0 ? 1 : -1, pointer)
    },
    [applyZoom]
  )

  const getPointer = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
    return groupRef.current?.getRelativePointerPosition() ?? null
  }, [])

  const commitCircle = useCallback(
    (x: number, y: number, r: number) => {
      if (r < MIN_RADIUS) return
      const geo = bboxFromCircle(x, y, r, fitScale.x, fitScale.y)
      onAddAnnotation({ shape: 'circle', label, ...geo })
    },
    [label, onAddAnnotation, fitScale]
  )

  const commitRect = useCallback(
    (rect: { x: number; y: number; width: number; height: number }) => {
      if (rect.width < MIN_STROKE || rect.height < MIN_STROKE) return
      const geo = bboxFromRect(rect, fitScale.x, fitScale.y)
      onAddAnnotation({ shape: 'rect', label, ...geo })
    },
    [label, onAddAnnotation, fitScale]
  )

  const commitBrush = useCallback(
    (points: number[]) => {
      if (points.length < 4) return
      const geo = bboxFromPoints(points, fitScale.x, fitScale.y)
      if (geo.origRadius < MIN_RADIUS) return
      onAddAnnotation({ shape: 'brush', label, ...geo })
    },
    [label, onAddAnnotation, fitScale]
  )

  const finishDrawing = useCallback(
    (e?: Konva.KonvaEventObject<MouseEvent>) => {
      if (!drawing || !startPos.current || tool === 'pan') return
      const pos = e ? getPointer(e) : null

      if (tool === 'circle' && pos) {
        const r = Math.hypot(pos.x - startPos.current.x, pos.y - startPos.current.y)
        commitCircle(startPos.current.x, startPos.current.y, r)
      } else if (tool === 'rect' && previewRect) {
        commitRect(previewRect)
      } else if (tool === 'brush' && brushStroke.length >= 4) {
        commitBrush(brushStroke)
      }

      setDrawing(false)
      setPreviewCircle(null)
      setPreviewRect(null)
      setBrushStroke([])
      startPos.current = null
    },
    [drawing, tool, previewRect, brushStroke, getPointer, commitCircle, commitRect, commitBrush]
  )

  const handleMouseDown = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      if (!image || tool === 'pan') return
      if (e.target.name()?.startsWith('det-hit')) return
      const pos = getPointer(e)
      if (!pos) return
      startPos.current = pos
      setDrawing(true)
      if (tool === 'brush') setBrushStroke([pos.x, pos.y])
    },
    [image, tool, getPointer]
  )

  const handleMouseMove = useCallback(
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      if (!drawing || !startPos.current || tool === 'pan') return
      const pos = getPointer(e)
      if (!pos) return

      if (tool === 'circle') {
        setPreviewCircle({
          x: startPos.current.x,
          y: startPos.current.y,
          r: Math.hypot(pos.x - startPos.current.x, pos.y - startPos.current.y),
        })
      } else if (tool === 'rect') {
        const x = Math.min(startPos.current.x, pos.x)
        const y = Math.min(startPos.current.y, pos.y)
        setPreviewRect({
          x,
          y,
          width: Math.abs(pos.x - startPos.current.x),
          height: Math.abs(pos.y - startPos.current.y),
        })
      } else if (tool === 'brush') {
        setBrushStroke((prev) => [...prev, pos.x, pos.y])
      }
    },
    [drawing, tool, getPointer]
  )

  if (!imageUrl) return null

  const cursor = tool === 'pan' ? 'grab' : 'crosshair'

  return (
    <div className={styles.wrap}>
      <div className={styles.zoomBar}>
        <button type="button" className={styles.zoomBtn} onClick={() => applyZoom(1)} aria-label="Yakınlaştır">
          +
        </button>
        <button type="button" className={styles.zoomBtn} onClick={() => applyZoom(-1)} aria-label="Uzaklaştır">
          −
        </button>
        <button type="button" className={styles.zoomBtn} onClick={resetView}>
          %{Math.round(zoom * 100)}
        </button>
        <span className={styles.zoomHint}>Tekerlek · Pan aracı</span>
      </div>

      <Stage
        width={containerSize.width}
        height={containerSize.height}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={finishDrawing}
        onMouseLeave={() => finishDrawing()}
        onClick={(e) => {
          if (e.target === e.target.getStage()) onSelectDetection(null)
        }}
        style={{ cursor }}
      >
        <Layer>
          <Group
            ref={groupRef}
            x={groupPos.x}
            y={groupPos.y}
            scaleX={zoom}
            scaleY={zoom}
            draggable={tool === 'pan'}
            onDragEnd={(e) => setGroupPos({ x: e.target.x(), y: e.target.y() })}
          >
            {image && <KonvaImage image={image} width={stageSize.width} height={stageSize.height} />}

            {detections.map((d, i) => {
              const [ox, oy, ow, oh] = d.bbox
              const x = ox * fitScale.x
              const y = oy * fitScale.y
              const w = ow * fitScale.x
              const h = oh * fitScale.y
              const color = colorForLabel(d.label)
              const pct = Math.round(d.confidence * 100)
              const selected = selectedDetectionIdx === i
              return (
                <Group
                  key={`det-${i}`}
                  onClick={(e) => {
                    e.cancelBubble = true
                    onSelectDetection(i)
                  }}
                >
                  <Rect
                    name={`det-hit-${i}`}
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    stroke={color}
                    strokeWidth={selected ? 3.5 : 2}
                    fill={selected ? `${color}45` : `${color}18`}
                  />
                  <Rect
                    x={x}
                    y={Math.max(0, y - 18)}
                    width={Math.max(w, 72)}
                    height={16}
                    fill={color}
                    cornerRadius={3}
                    listening={false}
                  />
                  <Text
                    x={x + 4}
                    y={Math.max(0, y - 16)}
                    text={`#${i + 1} ${d.label} ${pct}%`}
                    fontSize={11}
                    fill="white"
                    listening={false}
                  />
                </Group>
              )
            })}

            {annotations.map((ann) => (
              <AnnotationShape key={ann.id} ann={ann} />
            ))}

            {previewCircle && (
              <Circle
                x={previewCircle.x}
                y={previewCircle.y}
                radius={previewCircle.r}
                stroke="#64748b"
                strokeWidth={1.5}
                dash={[5, 3]}
              />
            )}
            {previewRect && (
              <Rect
                {...previewRect}
                stroke="#64748b"
                strokeWidth={1.5}
                dash={[5, 3]}
                fill="rgba(13, 148, 136, 0.12)"
              />
            )}
            {brushStroke.length > 0 && (
              <Line
                points={brushStroke}
                stroke="#0d9488"
                strokeWidth={brushSize}
                lineCap="round"
                lineJoin="round"
                tension={0.3}
              />
            )}
          </Group>
        </Layer>
      </Stage>
    </div>
  )
}

function AnnotationShape({ ann }: { ann: Annotation }) {
  let tagX = 0
  let tagY = 0

  if (ann.shape === 'circle' && ann.x != null && ann.y != null && ann.radius != null) {
    tagX = ann.x - 44
    tagY = ann.y - ann.radius - 22
  } else if (ann.shape === 'rect' && ann.rect) {
    tagX = ann.rect.x
    tagY = ann.rect.y - 22
  } else if (ann.shape === 'brush' && ann.points && ann.points.length >= 4) {
    const b = brushBounds(ann.points)
    tagX = b.x
    tagY = b.y - 22
  }

  return (
    <React.Fragment>
      {ann.shape === 'circle' && ann.x != null && ann.y != null && ann.radius != null && (
        <Circle x={ann.x} y={ann.y} radius={ann.radius} stroke={ann.color} strokeWidth={2.5} fill={`${ann.color}18`} />
      )}
      {ann.shape === 'rect' && ann.rect && (
        <Rect {...ann.rect} stroke={ann.color} strokeWidth={2.5} fill={`${ann.color}18`} />
      )}
      {ann.shape === 'brush' && ann.points && ann.points.length >= 4 && (
        <>
          <Line points={ann.points} stroke={ann.color} strokeWidth={3} lineCap="round" lineJoin="round" tension={0.3} />
          <Rect {...brushBounds(ann.points)} stroke={ann.color} strokeWidth={1} dash={[4, 3]} />
        </>
      )}
      <Rect x={tagX} y={tagY} width={88} height={18} fill={ann.color} cornerRadius={4} listening={false} />
      <Text
        x={tagX}
        y={tagY}
        width={88}
        height={18}
        text={ann.label}
        fontSize={11}
        fill="white"
        align="center"
        verticalAlign="middle"
        listening={false}
      />
    </React.Fragment>
  )
}
