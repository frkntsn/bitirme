import React, { useState, useRef, useCallback } from 'react'
import ImageCanvas from './components/ImageCanvas'
import AnnotationSidebar from './components/AnnotationSidebar'
import { useAnnotation } from './hooks/useAnnotation'
import { infer, resetMemory, segment, sendFeedback } from './api/client'
import { SegmentResult, DetectedRegion } from './types'
import styles from './App.module.css'

export default function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageBase64, setImageBase64] = useState<string | null>(null)
  const [label, setLabel] = useState('')
  const [scale, setScale] = useState({ x: 1, y: 1 })
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<SegmentResult[]>([])
  const [detections, setDetections] = useState<DetectedRegion[]>([])
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [correctedLabels, setCorrectedLabels] = useState<Record<number, string>>({})
  const containerRef = useRef<HTMLDivElement>(null)

  const { annotations, addAnnotation, removeAnnotation, clearAll } = useAnnotation()

  const loadFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = (e) => {
      const data = e.target?.result as string
      setImageUrl(data)
      setImageBase64(data.split(',')[1])
      setResults([])
      setDetections([])
      setError(null)
      setSuccessMessage(null)
      setCorrectedLabels({})
      clearAll()
    }
    reader.readAsDataURL(file)
  }, [clearAll])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (file) loadFile(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files[0]; if (file) loadFile(file)
  }

  const handleAddAnnotation = useCallback(
    (x: number, y: number, radius: number) => {
      addAnnotation(x, y, radius, label, scale.x, scale.y)
    }, [addAnnotation, label, scale]
  )

  const handleSend = async () => {
    if (!imageBase64 || annotations.length === 0) return
    setLoading(true)
    setError(null)
    setSuccessMessage(null)
    try {
      const res = await segment(imageBase64, annotations)
      setResults(res.results)
      setDetections(res.detections)
    } catch (err: any) {
      setError(err.message || 'Bilinmeyen hata')
    } finally {
      setLoading(false)
    }
  }

  const handleScanOnly = async () => {
    if (!imageBase64) return
    setLoading(true)
    setError(null)
    setSuccessMessage(null)
    try {
      const res = await infer(imageBase64)
      setResults(res.results)
      setDetections(res.detections)
    } catch (err: any) {
      setError(err.message || 'Bilinmeyen hata')
    } finally {
      setLoading(false)
    }
  }

  const handleResetLearnedMemory = async () => {
    setLoading(true)
    setError(null)
    setSuccessMessage(null)
    try {
      const res = await resetMemory()
      setResults([])
      setDetections([])
      setCorrectedLabels({})
      setSuccessMessage(
        `Öğrenilen sınıflar ve disk hafızası silindi. (bilinen sınıf: ${res.known_classes.length}, buffer: ${res.buffer_size})`
      )
    } catch (err: any) {
      setError(err.message || 'Sıfırlama hatası')
    } finally {
      setLoading(false)
    }
  }

  const handleDetectionFeedback = async (idx: number, accepted: boolean) => {
    const d = detections[idx]
    if (!d) return
    try {
      const feedbackRes = await sendFeedback([
        {
          label: d.label,
          predicted_label: d.label,
          accepted,
          corrected_label: correctedLabels[idx]?.trim() || undefined,
          crop_b64: d.crop_b64,
        },
      ])
      setError(`Geri bildirim işlendi. update=${feedbackRes.updated_count}, skip=${feedbackRes.skipped_count}`)
    } catch (err: any) {
      setError(err.message || 'Feedback hatası')
    }
  }

  const containerSize = {
    width: containerRef.current?.clientWidth ?? 700,
    height: containerRef.current?.clientHeight ?? 520,
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.title}>OCL Annotation</h1>
        <span className={styles.subtitle}>Fotoğraf yükle · Daire çiz · Etiketle · Tüm görseli tara</span>
        {detections.length > 0 && (
          <span className={styles.detectionBadge}>{detections.length} eşleşme bulundu</span>
        )}
      </header>

      <main className={styles.main}>
        <div
          ref={containerRef}
          className={`${styles.canvasWrap} ${dragging ? styles.dragOver : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          {!imageUrl ? (
            <div className={styles.placeholder}>
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="#bbb" strokeWidth="1.5">
                <rect x="4" y="4" width="40" height="40" rx="6" />
                <circle cx="16" cy="16" r="4" />
                <path d="M4 32l10-10 8 8 6-6 16 14" />
              </svg>
              <p>Fotoğraf yükle veya buraya sürükle</p>
              <p className={styles.placeholderSub}>patoloji · mikroskop · röntgen · genel fotoğraf</p>
              <label className={styles.uploadBtn}>
                Dosya Seç
                <input type="file" accept="image/*" onChange={handleFileInput} style={{ display: 'none' }} />
              </label>
            </div>
          ) : (
            <ImageCanvas
              imageUrl={imageUrl}
              annotations={annotations}
              detections={detections}
              onAddAnnotation={handleAddAnnotation}
              containerSize={containerSize}
              onScaleChange={(x, y) => setScale({ x, y })}
            />
          )}
        </div>

        <AnnotationSidebar
          label={label}
          onLabelChange={setLabel}
          annotations={annotations}
          onRemove={removeAnnotation}
          onClearAll={clearAll}
          onSend={handleSend}
          onScanOnly={handleScanOnly}
          onDetectionFeedback={handleDetectionFeedback}
          correctedLabels={correctedLabels}
          onCorrectedLabelChange={(idx, value) =>
            setCorrectedLabels((prev) => ({ ...prev, [idx]: value }))
          }
          loading={loading}
          results={results}
          detections={detections}
          error={error}
          successMessage={successMessage}
          onResetLearnedMemory={handleResetLearnedMemory}
        />
      </main>
    </div>
  )
}
