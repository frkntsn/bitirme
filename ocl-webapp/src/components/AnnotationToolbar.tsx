import { AnnotationTool } from '../types'
import styles from './AnnotationToolbar.module.css'

interface Props {
  tool: AnnotationTool
  onToolChange: (t: AnnotationTool) => void
  brushSize: number
  onBrushSizeChange: (n: number) => void
}

const TOOLS: { id: AnnotationTool; label: string; hint: string }[] = [
  { id: 'rect', label: 'Kutu', hint: 'Dikdörtgen — SAM kutu ipucu (en iyi)' },
  { id: 'circle', label: 'Daire', hint: 'Merkez nokta ipucu' },
  { id: 'brush', label: 'Fırça', hint: 'Serbest çizim — SAM kutu ipucu' },
  { id: 'pan', label: 'Pan', hint: 'Görüntüyü sürükle · tekerlek ile zoom' },
]

export default function AnnotationToolbar({
  tool,
  onToolChange,
  brushSize,
  onBrushSizeChange,
}: Props) {
  const activeHint = TOOLS.find((t) => t.id === tool)?.hint ?? ''

  return (
    <div className={styles.toolbar}>
      <span className={styles.label}>Araç</span>
      <div className={styles.tools} role="group" aria-label="Annotasyon aracı">
        {TOOLS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`${styles.toolBtn} ${tool === t.id ? styles.active : ''}`}
            onClick={() => onToolChange(t.id)}
            title={t.hint}
            aria-pressed={tool === t.id}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tool === 'brush' && (
        <label className={styles.brushRow}>
          <span>Kalınlık</span>
          <input
            type="range"
            min={2}
            max={24}
            value={brushSize}
            onChange={(e) => onBrushSizeChange(Number(e.target.value))}
          />
        </label>
      )}
      <p className={styles.hint}>{activeHint}</p>
    </div>
  )
}
