import React from 'react'
import { Annotation, SegmentResult, DetectedRegion } from '../types'
import styles from './AnnotationSidebar.module.css'

const QUICK_LABELS = ['tümör', 'normal', 'hücre', 'doku', 'lezyon']

interface Props {
  label: string
  onLabelChange: (v: string) => void
  annotations: Annotation[]
  onRemove: (id: string) => void
  onClearAll: () => void
  onSend: () => void
  onScanOnly: () => void
  onDetectionFeedback: (idx: number, accepted: boolean) => void
  correctedLabels: Record<number, string>
  onCorrectedLabelChange: (idx: number, value: string) => void
  loading: boolean
  results: SegmentResult[]
  detections: DetectedRegion[]
  error: string | null
  successMessage: string | null
  onResetLearnedMemory: () => void
}

export default function AnnotationSidebar({
  label, onLabelChange, annotations, onRemove,
  onClearAll, onSend, onScanOnly, onDetectionFeedback, correctedLabels, onCorrectedLabelChange,
  loading, results, detections, error, successMessage, onResetLearnedMemory,
}: Props) {
  return (
    <aside className={styles.sidebar}>

      <div className={styles.card}>
        <h3 className={styles.cardTitle}>Etiket</h3>
        <input
          className={styles.input} type="text" value={label}
          onChange={(e) => onLabelChange(e.target.value)}
          placeholder="sınıf adı gir..."
          onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        />
        <div className={styles.quickLabels}>
          <span className={styles.quickHint}>Hızlı:</span>
          {QUICK_LABELS.map((l) => (
            <button key={l} className={styles.chip} onClick={() => onLabelChange(l)}>{l}</button>
          ))}
        </div>
      </div>

      <div className={`${styles.card} ${styles.listCard}`}>
        <div className={styles.listHeader}>
          <h3 className={styles.cardTitle}>Annotasyonlar ({annotations.length})</h3>
          {annotations.length > 0 && (
            <button className={styles.textBtn} onClick={onClearAll}>temizle</button>
          )}
        </div>
        <div className={styles.list}>
          {annotations.length === 0 ? (
            <p className={styles.empty}>Daire çizerek örnek göster</p>
          ) : (
            annotations.map((ann) => (
              <div key={ann.id} className={styles.annItem}>
                <div className={styles.dot} style={{ background: ann.color }} />
                <span className={styles.annLabel}>{ann.label}</span>
                <span className={styles.annMeta}>{Math.round(ann.radius)}px</span>
                <button className={styles.delBtn} onClick={() => onRemove(ann.id)}>×</button>
              </div>
            ))
          )}
        </div>
      </div>

      <button
        className={styles.sendBtn}
        disabled={annotations.length === 0 || loading}
        onClick={onSend}
      >
        {loading ? 'Tüm görsel taranıyor...' : `Gönder + Tüm Görseli Tara`}
      </button>
      <button
        className={styles.sendBtn}
        disabled={loading}
        onClick={onScanOnly}
        style={{ marginTop: 8, background: '#374151' }}
      >
        {loading ? 'Taranıyor...' : 'Sadece Tara (Annotasyonsuz)'}
      </button>

      <button
        type="button"
        className={styles.dangerBtn}
        disabled={loading}
        onClick={() => {
          if (
            window.confirm(
              'Sunucudaki öğrenilen tüm sınıflar ve replay buffer silinecek (diskteki hafıza dahil). Emin misiniz?'
            )
          ) {
            onResetLearnedMemory()
          }
        }}
      >
        Öğrenilen sınıfları unut
      </button>

      {successMessage && <div className={styles.successBox}>{successMessage}</div>}
      {error && <div className={styles.errorBox}>{error}</div>}

      {/* Tespit özeti */}
      {detections.length > 0 && (
        <div className={styles.resultsCard}>
          <h3 className={styles.cardTitle}>Tespitler ({detections.length})</h3>
          {/* Sınıf bazında grupla */}
          {Object.entries(
            detections.reduce((acc, d) => {
              acc[d.label] = (acc[d.label] || 0) + 1
              return acc
            }, {} as Record<string, number>)
          ).map(([lbl, count]) => (
            <div key={lbl} className={styles.resultItem}>
              <div className={styles.resultRow}>
                <span className={styles.resultClass}>{lbl}</span>
                <span className={styles.resultCount}>{count} bölge</span>
              </div>
              <div className={styles.confidence}>
                <div
                  className={styles.confidenceBar}
                  style={{
                    width: `${Math.round(
                      (detections.filter(d => d.label === lbl).reduce((s, d) => s + d.confidence, 0) / count) * 100
                    )}%`
                  }}
                />
                <span className={styles.confidenceText}>
                  ort. %{Math.round(
                    (detections.filter(d => d.label === lbl).reduce((s, d) => s + d.confidence, 0) / count) * 100
                  )}
                </span>
              </div>
            </div>
          ))}

          {/* Tek tek geri bildirim */}
          <div style={{ marginTop: 10 }}>
            {detections.map((d, i) => (
              <div key={`fb-${i}`} className={styles.resultItem}>
                <div className={styles.resultRow}>
                  <span className={styles.resultClass}>{d.label}</span>
                  <span className={styles.confidenceText}>%{Math.round(d.confidence * 100)}</span>
                </div>
                <div className={styles.resultRow} style={{ marginTop: 6, gap: 8 }}>
                  <button className={styles.chip} onClick={() => onDetectionFeedback(i, true)}>Doğru</button>
                  <button className={styles.chip} onClick={() => onDetectionFeedback(i, false)}>Yanlış</button>
                </div>
                <input
                  className={styles.input}
                  style={{ marginTop: 6 }}
                  placeholder="Yanlışsa doğrusu (opsiyonel)"
                  value={correctedLabels[i] || ''}
                  onChange={(e) => onCorrectedLabelChange(i, e.target.value)}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Annotation başına sonuç */}
      {results.length > 0 && (
        <div className={styles.resultsCard}>
          <h3 className={styles.cardTitle}>Annotation sonuçları</h3>
          {results.map((r, i) => (
            <div key={i} className={styles.resultItem}>
              <div className={styles.resultRow}>
                <span className={styles.resultLabel}>{r.label}</span>
                <span className={styles.resultArrow}>→</span>
                <span className={styles.resultClass}>{r.predicted_class}</span>
                <span className={styles.confidenceText}>%{Math.round(r.confidence * 100)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </aside>
  )
}
