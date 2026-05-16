import { DetectedRegion } from '../types'
import styles from './DetectionInspectPanel.module.css'

interface Props {
  index: number
  total: number
  detection: DetectedRegion
  correctedLabel: string
  onCorrectedLabelChange: (value: string) => void
  onFeedback: (accepted: boolean) => void
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
}

export default function DetectionInspectPanel({
  index,
  total,
  detection,
  correctedLabel,
  onCorrectedLabelChange,
  onFeedback,
  onClose,
  onPrev,
  onNext,
}: Props) {
  const pct = Math.round(detection.confidence * 100)
  const cropSrc = detection.crop_b64
    ? detection.crop_b64.startsWith('data:')
      ? detection.crop_b64
      : `data:image/png;base64,${detection.crop_b64}`
    : null

  return (
    <div className={styles.backdrop} onClick={onClose} role="presentation">
      <div
        className={styles.panel}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-labelledby="det-panel-title"
      >
        <header className={styles.header}>
          <div>
            <h2 id="det-panel-title" className={styles.title}>
              Tespit #{index + 1}
            </h2>
            <p className={styles.sub}>
              {index + 1} / {total} · canvas’taki kutuya karşılık gelir
            </p>
          </div>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Kapat">
            ×
          </button>
        </header>

        <div className={styles.body}>
          {cropSrc ? (
            <img className={styles.crop} src={cropSrc} alt={`Kırpım: ${detection.label}`} />
          ) : (
            <div className={styles.cropPlaceholder}>Önizleme yok</div>
          )}

          <div className={styles.meta}>
            <span className={styles.predLabel}>{detection.label}</span>
            <span className={styles.conf}>%{pct} güven</span>
          </div>

          <p className={styles.question}>Bu tespit doğru mu?</p>

          <div className={styles.actions}>
            <button type="button" className={styles.okBtn} onClick={() => onFeedback(true)}>
              Doğru — bu sınıf
            </button>
            <button type="button" className={styles.badBtn} onClick={() => onFeedback(false)}>
              Yanlış
            </button>
          </div>

          <label className={styles.field}>
            <span>Yanlışsa doğru etiket</span>
            <input
              type="text"
              value={correctedLabel}
              onChange={(e) => onCorrectedLabelChange(e.target.value)}
              placeholder={`Örn. farklı sınıf adı (şu an: ${detection.label})`}
            />
          </label>
        </div>

        {(onPrev || onNext) && (
          <footer className={styles.footer}>
            <button type="button" disabled={!onPrev} onClick={onPrev}>
              ← Önceki
            </button>
            <button type="button" disabled={!onNext} onClick={onNext}>
              Sonraki →
            </button>
          </footer>
        )}
      </div>
    </div>
  )
}
