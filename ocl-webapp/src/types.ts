export interface Annotation {
  id: string
  x: number
  y: number
  radius: number
  label: string
  color: string
  origX: number
  origY: number
  origRadius: number
}

export interface SegmentResult {
  label: string
  predicted_class: string
  confidence: number
  crop_b64?: string
  mask_area?: number
}

export interface DetectedRegion {
  bbox: [number, number, number, number]  // [x, y, w, h] orijinal koordinat
  label: string
  confidence: number
  crop_b64?: string
  mask_area?: number
}

export interface SegmentResponse {
  results: SegmentResult[]
  detections: DetectedRegion[]
  model_updated: boolean
}

export interface FeedbackItem {
  label: string
  accepted: boolean
  predicted_label?: string
  corrected_label?: string
  crop_b64?: string
}

export interface FeedbackResponse {
  updated_count: number
  skipped_count: number
}

export interface ResetMemoryResponse {
  ok: boolean
  buffer_size: number
  known_classes: string[]
}
