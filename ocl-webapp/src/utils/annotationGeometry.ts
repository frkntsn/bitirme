export function bboxFromPoints(
  points: number[],
  scaleX: number,
  scaleY: number
): { origX: number; origY: number; origRadius: number; origBbox: [number, number, number, number]; origPoints: number[] } {
  const origPoints: number[] = []
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < points.length; i += 2) {
    const ox = points[i] / scaleX
    const oy = points[i + 1] / scaleY
    origPoints.push(ox, oy)
    xs.push(ox)
    ys.push(oy)
  }
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const w = maxX - minX
  const h = maxY - minY
  return {
    origX: Math.round(minX + w / 2),
    origY: Math.round(minY + h / 2),
    origRadius: Math.round(Math.max(w, h) / 2),
    origBbox: [Math.round(minX), Math.round(minY), Math.round(w), Math.round(h)],
    origPoints,
  }
}

export function bboxFromRect(
  rect: { x: number; y: number; width: number; height: number },
  scaleX: number,
  scaleY: number
): { origX: number; origY: number; origRadius: number; origBbox: [number, number, number, number] } {
  const w = rect.width / scaleX
  const h = rect.height / scaleY
  const ox = rect.x / scaleX
  const oy = rect.y / scaleY
  return {
    origX: Math.round(ox + w / 2),
    origY: Math.round(oy + h / 2),
    origRadius: Math.round(Math.max(w, h) / 2),
    origBbox: [Math.round(ox), Math.round(oy), Math.round(w), Math.round(h)],
  }
}

export function bboxFromCircle(
  x: number,
  y: number,
  radius: number,
  scaleX: number,
  scaleY: number
): { origX: number; origY: number; origRadius: number; origBbox: [number, number, number, number] } {
  const ox = x / scaleX
  const oy = y / scaleY
  const r = radius / Math.min(scaleX, scaleY)
  return {
    origX: Math.round(ox),
    origY: Math.round(oy),
    origRadius: Math.round(r),
    origBbox: [Math.round(ox - r), Math.round(oy - r), Math.round(2 * r), Math.round(2 * r)],
  }
}

export function shapeLabel(shape: string): string {
  switch (shape) {
    case 'rect':
      return 'kutu'
    case 'brush':
      return 'fırça'
    default:
      return 'daire'
  }
}
