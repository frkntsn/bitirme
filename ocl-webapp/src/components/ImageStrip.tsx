import { ImageSession } from '../types'
import styles from './ImageStrip.module.css'

interface Props {
  sessions: ImageSession[]
  activeId: string | null
  onSelect: (id: string) => void
  onNewImage: () => void
  onRemoveActive: () => void
  hasActive: boolean
}

export default function ImageStrip({
  sessions,
  activeId,
  onSelect,
  onNewImage,
  onRemoveActive,
  hasActive,
}: Props) {
  return (
    <div className={styles.strip}>
      <div className={styles.actions}>
        <button type="button" className={styles.primaryBtn} onClick={onNewImage}>
          + Yeni görsel
        </button>
        {hasActive && (
          <button type="button" className={styles.ghostBtn} onClick={onRemoveActive}>
            Görseli kaldır
          </button>
        )}
      </div>

      {sessions.length > 0 && (
        <div className={styles.thumbs}>
          {sessions.map((s, i) => (
            <button
              key={s.id}
              type="button"
              className={`${styles.thumb} ${s.id === activeId ? styles.thumbActive : ''}`}
              onClick={() => onSelect(s.id)}
              title={s.name}
            >
              <img src={s.dataUrl} alt="" />
              <span className={styles.thumbLabel}>
                {i + 1}. {s.name.length > 14 ? `${s.name.slice(0, 12)}…` : s.name}
              </span>
              {s.annotations.length > 0 && (
                <span className={styles.thumbBadge}>{s.annotations.length} örnek</span>
              )}
            </button>
          ))}
        </div>
      )}

      <p className={styles.hint}>
        Her görselin çizimleri ayrı saklanır. Görsel değiştirdikten sonra o görsel için &quot;Bu görseli öğret + tara&quot; ile ayrı öğretin.
      </p>
    </div>
  )
}
