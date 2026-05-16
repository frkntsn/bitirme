import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import ImageCanvas from './components/ImageCanvas'
import AnnotationSidebar from './components/AnnotationSidebar'
import AnnotationToolbar from './components/AnnotationToolbar'
import LeftNavPanel from './components/LeftNavPanel'
import DetectionInspectPanel from './components/DetectionInspectPanel'
import { useImageSessions } from './hooks/useImageSessions'
import { useBackendHealth } from './hooks/useBackendHealth'
import { useContainerSize } from './hooks/useContainerSize'
import { infer, resetMemory, segment, sendFeedback } from './api/client'
import { AnnotationInput, AnnotationTool, DetectedRegion } from './types'
import { inputToStored, projectAnnotations } from './utils/annotationProjection'
import { useMediaQuery } from './hooks/useMediaQuery'
import { useTheme, Theme } from './hooks/useTheme'
import { APP_ICON_URL } from './config/branding'
import styles from './App.module.css'

function filterDetections(dets: DetectedRegion[], allowed: Set<string>) {
  if (allowed.size === 0) return []
  return dets.filter((d) => allowed.has(d.label))
}

export default function App() {
  const [label, setLabel] = useState('')
  const [tool, setTool] = useState<AnnotationTool>('rect')
  const [brushSize, setBrushSize] = useState(8)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [correctedLabels, setCorrectedLabels] = useState<Record<number, string>>({})
  const [selectedDetectionIdx, setSelectedDetectionIdx] = useState<number | null>(null)
  const [fitScale, setFitScale] = useState({ x: 1, y: 1 })

  const [navOpen, setNavOpen] = useState(false)
  const [navCollapsed, setNavCollapsed] = useState(false)
  const [rightOpen, setRightOpen] = useState(true)
  const [scanClassFilter, setScanClassFilter] = useState<Set<string>>(new Set())

  const containerRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const containerSize = useContainerSize(containerRef)

  const {
    sessions,
    active,
    activeId,
    addFromFile,
    selectSession,
    removeActive,
    patchActive,
    setSessions,
  } = useImageSessions()

  const { health } = useBackendHealth()
  const { theme, setTheme } = useTheme()
  const knownClasses = health?.known_classes ?? []

  useEffect(() => {
    if (knownClasses.length === 0) {
      setScanClassFilter(new Set())
      return
    }
    setScanClassFilter((prev) => {
      if (prev.size === 0) return new Set(knownClasses)
      const next = new Set<string>()
      for (const c of knownClasses) {
        if (prev.has(c)) next.add(c)
      }
      if (next.size === 0) return new Set(knownClasses)
      return next
    })
  }, [knownClasses.join('|')])

  const imageUrl = active?.dataUrl ?? null
  const imageBase64 = active?.base64 ?? null
  const storedAnnotations = active?.annotations ?? []
  const rawDetections = active?.detections ?? []
  const results = active?.results ?? []

  const visibleDetections = useMemo(
    () => filterDetections(rawDetections, scanClassFilter),
    [rawDetections, scanClassFilter]
  )

  const displayAnnotations = useMemo(
    () => projectAnnotations(storedAnnotations, fitScale.x, fitScale.y),
    [storedAnnotations, fitScale]
  )

  useEffect(() => {
    setSelectedDetectionIdx(null)
    setCorrectedLabels({})
  }, [activeId, scanClassFilter])

  const loadFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return
      const reader = new FileReader()
      reader.onload = (e) => {
        const data = e.target?.result as string
        const img = new window.Image()
        img.onload = () => {
          addFromFile(file, data, data.split(',')[1], img.width, img.height)
          setError(null)
          setInfo(`"${file.name}" yüklendi.`)
          setNavOpen(true)
        }
        img.src = data
      }
      reader.readAsDataURL(file)
    },
    [addFromFile]
  )

  const openFilePicker = () => fileInputRef.current?.click()

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) loadFile(file)
    e.target.value = ''
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) loadFile(file)
  }

  const handleRemoveImage = () => {
    if (!active) return
    if (!window.confirm(`"${active.name}" kaldırılsın mı?`)) return
    removeActive()
    setInfo('Görsel kaldırıldı.')
  }

  const handleScaleChange = useCallback((sx: number, sy: number) => {
    setFitScale({ x: sx, y: sy })
  }, [])

  const handleAddAnnotation = useCallback(
    (input: AnnotationInput) => {
      if (!activeId) return
      const stored = inputToStored({ ...input, label: input.label || label })
      patchActive({ annotations: [...storedAnnotations, stored] })
    },
    [activeId, label, patchActive, storedAnnotations]
  )

  const handleRemoveAnnotation = useCallback(
    (id: string) => {
      patchActive({ annotations: storedAnnotations.filter((a) => a.id !== id) })
    },
    [patchActive, storedAnnotations]
  )

  const handleClearAnnotations = useCallback(() => {
    patchActive({ annotations: [] })
  }, [patchActive])

  const handleSend = async () => {
    if (!imageBase64 || !activeId || storedAnnotations.length === 0) return
    setLoading(true)
    setError(null)
    setInfo(null)
    try {
      const res = await segment(imageBase64, storedAnnotations)
      patchActive({ results: res.results, detections: res.detections })
      setSelectedDetectionIdx(null)
      setInfo(
        res.model_updated
          ? `"${active?.name}" üzerinde model güncellendi.`
          : 'Tarama tamamlandı.'
      )
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Bilinmeyen hata')
    } finally {
      setLoading(false)
    }
  }

  const handleScanOnly = async () => {
    if (!imageBase64 || !activeId) return
    if (scanClassFilter.size === 0) {
      setError('Tarama için en az bir sınıf seç.')
      return
    }
    setLoading(true)
    setError(null)
    setInfo(null)
    try {
      const res = await infer(imageBase64)
      const filtered = filterDetections(res.detections, scanClassFilter)
      patchActive({ results: res.results, detections: res.detections })
      setSelectedDetectionIdx(null)
      setInfo(
        filtered.length > 0
          ? `${filtered.length} tespit (seçili sınıflar) — kutuya tıkla.`
          : 'Seçili sınıflarda eşleşme yok.'
      )
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Bilinmeyen hata')
    } finally {
      setLoading(false)
    }
  }

  const handleResetLearnedMemory = async () => {
    setLoading(true)
    setError(null)
    try {
      await resetMemory()
      setSessions((prev) => prev.map((s) => ({ ...s, detections: [], results: [] })))
      setScanClassFilter(new Set())
      setInfo('Öğrenilen sınıflar silindi.')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Sıfırlama hatası')
    } finally {
      setLoading(false)
    }
  }

  const handleDetectionFeedback = async (visIdx: number, accepted: boolean) => {
    const d = visibleDetections[visIdx]
    if (!d) return
    try {
      const feedbackRes = await sendFeedback([
        {
          label: d.label,
          predicted_label: d.label,
          accepted,
          corrected_label: correctedLabels[visIdx]?.trim() || undefined,
          crop_b64: d.crop_b64,
        },
      ])
      setInfo(`Tespit kaydedildi (${feedbackRes.updated_count} güncelleme).`)
      setSelectedDetectionIdx(null)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Feedback hatası')
    }
  }

  const selectedDetection =
    selectedDetectionIdx != null ? visibleDetections[selectedDetectionIdx] : null

  const isTablet = useMediaQuery('(max-width: 1024px)')

  useEffect(() => {
    setNavOpen(!isTablet)
  }, [isTablet])

  return (
    <div className={styles.app}>
      <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileInput} hidden />

      <header className={styles.header}>
        <button
          type="button"
          className={styles.menuBtn}
          onClick={() => setNavOpen((o) => !o)}
          aria-label="Gezgin paneli"
        >
          ☰
        </button>
        <div className={styles.brand}>
          <div className={styles.brandThumb}>
            <img src={APP_ICON_URL} alt="" className={styles.brandThumbImg} />
          </div>
          <h1 className={styles.title}>Canvas</h1>
        </div>
        <div className={styles.headerActions}>
          <label className={styles.themeLabel}>
            <span className={styles.themeLabelText}>Tema</span>
            <select
              className={styles.themeSelect}
              value={theme}
              onChange={(e) => setTheme(e.target.value as Theme)}
              aria-label="Tema seçimi"
            >
              <option value="dark">Koyu</option>
              <option value="light">Açık</option>
            </select>
          </label>
          <button type="button" className={styles.btnPrimary} onClick={openFilePicker}>
            + Görsel
          </button>
          <button
            type="button"
            className={styles.btnGhost}
            onClick={() => setRightOpen((o) => !o)}
            aria-label="Araç paneli"
          >
            Araçlar
          </button>
        </div>
      </header>

      {navOpen && isTablet && (
        <button
          type="button"
          className={styles.backdrop}
          aria-label="Paneli kapat"
          onClick={() => setNavOpen(false)}
        />
      )}

      <div className={styles.body}>
        <LeftNavPanel
          open={navOpen}
          collapsed={navCollapsed && !isTablet}
          onToggleCollapse={() => setNavCollapsed((c) => !c)}
          onCloseMobile={() => setNavOpen(false)}
          sessions={sessions}
          activeId={activeId}
          onSelectImage={(id) => {
            selectSession(id)
            if (isTablet) setNavOpen(false)
          }}
          onNewImage={openFilePicker}
          onRemoveActive={handleRemoveImage}
          hasActive={!!active}
          knownClasses={knownClasses}
          scanClassFilter={scanClassFilter}
          onToggleScanClass={(c) => {
            setScanClassFilter((prev) => {
              const next = new Set(prev)
              if (next.has(c)) next.delete(c)
              else next.add(c)
              return next
            })
          }}
          onSelectAllScanClasses={() => setScanClassFilter(new Set(knownClasses))}
          onClearScanClassFilter={() => setScanClassFilter(new Set())}
          onResetMemory={handleResetLearnedMemory}
          loading={loading}
        />

        <main className={styles.main}>
          <section className={styles.stagePanel}>
            <div
              ref={containerRef}
              className={`${styles.canvasFrame} ${dragging ? styles.canvasDrag : ''}`}
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              {!imageUrl ? (
                <div className={styles.emptyState}>
                  <p className={styles.emptyTitle}>Görsel yükle</p>
                  <p className={styles.emptySub}>
                    Sol panelden görsel ve sınıf yönet · tablet için Araçlar
                  </p>
                  <button type="button" className={styles.btnPrimary} onClick={openFilePicker}>
                    Dosya seç
                  </button>
                </div>
              ) : (
                <>
                  {loading && (
                    <div className={styles.loading}>
                      <div className={styles.spinner} />
                      <span>İşleniyor…</span>
                    </div>
                  )}
                  <div className={styles.canvasInner}>
                    <ImageCanvas
                      imageUrl={imageUrl}
                      annotations={displayAnnotations}
                      detections={visibleDetections}
                      tool={tool}
                      brushSize={brushSize}
                      label={label}
                      selectedDetectionIdx={selectedDetectionIdx}
                      onSelectDetection={setSelectedDetectionIdx}
                      onAddAnnotation={handleAddAnnotation}
                      containerSize={containerSize}
                      onScaleChange={handleScaleChange}
                    />
                  </div>
                  {selectedDetection && selectedDetectionIdx != null && (
                    <DetectionInspectPanel
                      index={selectedDetectionIdx}
                      total={visibleDetections.length}
                      detection={selectedDetection}
                      correctedLabel={correctedLabels[selectedDetectionIdx] ?? ''}
                      onCorrectedLabelChange={(v) =>
                        setCorrectedLabels((p) => ({ ...p, [selectedDetectionIdx]: v }))
                      }
                      onFeedback={(ok) => handleDetectionFeedback(selectedDetectionIdx, ok)}
                      onClose={() => setSelectedDetectionIdx(null)}
                      onPrev={
                        selectedDetectionIdx > 0
                          ? () => setSelectedDetectionIdx(selectedDetectionIdx - 1)
                          : undefined
                      }
                      onNext={
                        selectedDetectionIdx < visibleDetections.length - 1
                          ? () => setSelectedDetectionIdx(selectedDetectionIdx + 1)
                          : undefined
                      }
                    />
                  )}
                </>
              )}
            </div>
            {imageUrl && (
              <AnnotationToolbar
                tool={tool}
                onToolChange={setTool}
                brushSize={brushSize}
                onBrushSizeChange={setBrushSize}
              />
            )}
          </section>
        </main>

        <aside className={`${styles.rightPanel} ${rightOpen ? styles.rightPanelOpen : ''}`}>
          <AnnotationSidebar
            label={label}
            onLabelChange={setLabel}
            annotations={storedAnnotations}
            onRemove={handleRemoveAnnotation}
            onClearAll={handleClearAnnotations}
            onSend={handleSend}
            onScanOnly={handleScanOnly}
            loading={loading}
            results={results}
            detections={visibleDetections}
            error={error}
            info={info}
            activeImageName={active?.name}
            selectedDetectionIdx={selectedDetectionIdx}
            onSelectDetection={setSelectedDetectionIdx}
            scanClassCount={scanClassFilter.size}
            knownClassCount={knownClasses.length}
          />
        </aside>
      </div>

      {rightOpen && isTablet && (
        <button
          type="button"
          className={styles.backdrop}
          aria-label="Araç panelini kapat"
          onClick={() => setRightOpen(false)}
        />
      )}
    </div>
  )
}
