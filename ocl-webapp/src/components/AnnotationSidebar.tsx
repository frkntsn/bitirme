import { StoredAnnotation, SegmentResult, DetectedRegion } from '../types'
import { shapeLabel } from '../utils/annotationGeometry'
import styles from './AnnotationSidebar.module.css'

const QUICK_LABELS = ['tümör', 'normal', 'hücre', 'doku', 'lezyon']

interface Props {
  label: string
  onLabelChange: (v: string) => void
  annotations: StoredAnnotation[]
  onRemove: (id: string) => void
  onClearAll: () => void
  onSend: () => void
  onScanOnly: () => void
  loading: boolean
  results: SegmentResult[]
  detections: DetectedRegion[]
  error: string | null
  info: string | null
  activeImageName?: string
  selectedDetectionIdx: number | null
  onSelectDetection: (idx: number | null) => void
  scanClassCount: number
  knownClassCount: number
}

export default function AnnotationSidebar({
  label,
  onLabelChange,
  annotations,
  onRemove,
  onClearAll,
  onSend,
  onScanOnly,
  loading,
  results,
  detections,
  error,
  info,
  activeImageName,
  selectedDetectionIdx,
  onSelectDetection,
  scanClassCount,
  knownClassCount,
}: Props) {
  return (
    <aside className={styles.sidebar}>
      {activeImageName && (
        <div className={styles.activeFile}>
          <span className={styles.activeFileLabel}>Aktif</span>
          <span className={styles.activeFileName}>{activeImageName}</span>
        </div>
      )}

      <div className={styles.card}>
        <h3 className={styles.cardTitle}>Sınıf etiketi</h3>
        <input
          className={styles.input}
          value={label}
          onChange={(e) => onLabelChange(e.target.value)}
          placeholder="Çizeceğin bölgenin etiketi..."
        />
        <div className={styles.chips}>
          {QUICK_LABELS.map((l) => (
            <button key={l} type="button" className={styles.chip} onClick={() => onLabelChange(l)}>
              {l}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardHead}>
          <h3 className={styles.cardTitle}>Örnekler</h3>
          <span className={styles.count}>{annotations.length}</span>
        </div>
        {annotations.length > 0 && (
          <button type="button" className={styles.linkBtn} onClick={onClearAll}>
            Temizle
          </button>
        )}
        <div className={styles.list}>
          {annotations.length === 0 ? (
            <p className={styles.empty}>Daire, kutu veya fırça ile bölge çiz</p>
          ) : (
            annotations.map((ann) => (
              <div key={ann.id} className={styles.annRow}>
                <span className={styles.dot} style={{ background: ann.color }} />
                <span className={styles.annLabel}>{ann.label}</span>
                <span className={styles.annMeta}>{shapeLabel(ann.shape)}</span>
                <button type="button" className={styles.delBtn} onClick={() => onRemove(ann.id)}>
                  ×
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <button
        type="button"
        className={styles.cta}
        disabled={annotations.length === 0 || loading}
        onClick={onSend}
      >
        {loading ? 'İşleniyor…' : 'Öğret + tara'}
      </button>
      <button
        type="button"
        className={styles.ctaSecondary}
        disabled={loading || knownClassCount === 0 || scanClassCount === 0}
        onClick={onScanOnly}
      >
        Sadece tara ({scanClassCount}/{knownClassCount})
      </button>

      {info && <div className={styles.info}>{info}</div>}
      {error && <div className={styles.error}>{error}</div>}

      {detections.length > 0 && (
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Tespitler</h3>
          <p className={styles.hint}>Kutuya tıkla → düzelt</p>
          <div className={styles.detList}>
            {detections.map((d, i) => (
              <button
                key={i}
                type="button"
                className={`${styles.detRow} ${selectedDetectionIdx === i ? styles.detActive : ''}`}
                onClick={() => onSelectDetection(i)}
              >
                <span className={styles.detNum}>#{i + 1}</span>
                <span className={styles.detLbl}>{d.label}</span>
                <span className={styles.detPct}>%{Math.round(d.confidence * 100)}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Öğrenme</h3>
          {results.map((r, i) => (
            <div key={i} className={styles.resultRow}>
              <span>{r.label}</span>
              <span>→ {r.predicted_class}</span>
              <span className={styles.pct}>%{Math.round(r.confidence * 100)}</span>
            </div>
          ))}
        </div>
      )}
    </aside>
  )
}
