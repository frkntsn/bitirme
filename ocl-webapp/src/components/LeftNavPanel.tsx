import { ImageSession } from '../types'
import styles from './LeftNavPanel.module.css'

interface Props {
  open: boolean
  collapsed: boolean
  onToggleCollapse: () => void
  onCloseMobile: () => void
  sessions: ImageSession[]
  activeId: string | null
  onSelectImage: (id: string) => void
  onNewImage: () => void
  onRemoveActive: () => void
  hasActive: boolean
  knownClasses: string[]
  scanClassFilter: Set<string>
  onToggleScanClass: (label: string) => void
  onSelectAllScanClasses: () => void
  onClearScanClassFilter: () => void
  onResetMemory: () => void
  loading: boolean
}

export default function LeftNavPanel({
  open,
  collapsed,
  onToggleCollapse,
  onCloseMobile,
  sessions,
  activeId,
  onSelectImage,
  onNewImage,
  onRemoveActive,
  hasActive,
  knownClasses,
  scanClassFilter,
  onToggleScanClass,
  onSelectAllScanClasses,
  onClearScanClassFilter,
  onResetMemory,
  loading,
}: Props) {
  return (
    <aside
      className={`${styles.panel} ${open ? styles.panelOpen : ''} ${collapsed ? styles.panelCollapsed : ''}`}
    >
      <div className={styles.panelInner}>
        <div className={styles.panelHead}>
          {!collapsed && <span className={styles.panelTitle}>Gezgin</span>}
          <button
            type="button"
            className={styles.iconBtn}
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Paneli genişlet' : 'Paneli daralt'}
          >
            {collapsed ? '›' : '‹'}
          </button>
          {!collapsed && (
            <button
              type="button"
              className={`${styles.iconBtn} ${styles.closeMobile}`}
              onClick={onCloseMobile}
              aria-label="Kapat"
            >
              ×
            </button>
          )}
        </div>

        {!collapsed && (
          <>
            <div className={styles.panelScroll}>
              <section className={styles.section}>
                <div className={styles.sectionHead}>
                  <h2 className={styles.sectionTitle}>Görseller</h2>
                  <button type="button" className={styles.miniBtn} onClick={onNewImage}>
                    + Ekle
                  </button>
                </div>
                {sessions.length === 0 ? (
                  <p className={styles.empty}>Henüz görsel yok</p>
                ) : (
                  <ul className={styles.imageList}>
                    {sessions.map((s, i) => (
                      <li key={s.id}>
                        <button
                          type="button"
                          className={`${styles.imageRow} ${s.id === activeId ? styles.imageRowActive : ''}`}
                          onClick={() => onSelectImage(s.id)}
                        >
                          <img src={s.previewUrl || s.dataUrl} alt="" className={styles.thumb} />
                          <span className={styles.imageMeta}>
                            <span className={styles.imageName}>
                              {i + 1}. {s.name}
                            </span>
                            <span className={styles.imageStats}>
                              {s.annotations.length} örnek
                              {s.detections.length > 0 && ` · ${s.detections.length} tespit`}
                            </span>
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                {hasActive && (
                  <button type="button" className={styles.ghostLink} onClick={onRemoveActive}>
                    Aktif görseli kaldır
                  </button>
                )}
              </section>

              <section className={styles.section}>
                <h2 className={styles.sectionTitle}>Öğrenilen sınıflar</h2>
                {knownClasses.length === 0 ? (
                  <p className={styles.empty}>Örnek çizip gönderince burada görünür</p>
                ) : (
                  <ul className={styles.classList}>
                    {knownClasses.map((c) => (
                      <li key={c} className={styles.classChip}>
                        <span className={styles.classDot} />
                        {c}
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section className={styles.section}>
                <div className={styles.sectionHead}>
                  <h2 className={styles.sectionTitle}>Tarama filtresi</h2>
                  <div className={styles.filterActions}>
                    <button type="button" className={styles.linkBtn} onClick={onSelectAllScanClasses}>
                      Tümü
                    </button>
                    <button type="button" className={styles.linkBtn} onClick={onClearScanClassFilter}>
                      Temizle
                    </button>
                  </div>
                </div>
                <p className={styles.hint}>Canvas’ta gösterilecek sınıflar</p>
                {knownClasses.length === 0 ? (
                  <p className={styles.empty}>Önce sınıf öğretin</p>
                ) : (
                  <ul className={styles.checkList}>
                    {knownClasses.map((c) => (
                      <li key={c}>
                        <label className={styles.checkRow}>
                          <input
                            type="checkbox"
                            checked={scanClassFilter.has(c)}
                            onChange={() => onToggleScanClass(c)}
                          />
                          <span>{c}</span>
                        </label>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </div>

            <div className={styles.panelFoot}>
              <button
                type="button"
                className={styles.dangerBtn}
                disabled={loading}
                onClick={() =>
                  window.confirm('Tüm öğrenilen sınıflar silinsin mi?') && onResetMemory()
                }
              >
                Hafızayı sıfırla
              </button>
            </div>
          </>
        )}

        {collapsed && (
          <div className={styles.rail}>
            <button type="button" className={styles.railBtn} onClick={onNewImage} title="Yeni görsel">
              +
            </button>
            <span className={styles.railCount}>{sessions.length}</span>
            <span className={styles.railCount}>{knownClasses.length}</span>
          </div>
        )}
      </div>
    </aside>
  )
}
