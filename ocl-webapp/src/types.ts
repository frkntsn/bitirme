export type AnnotationTool = 'circle' | 'rect' | 'brush' | 'pan'

export type AnnotationShape = AnnotationTool

/** Disk / oturumda saklanan — yalnızca orijinal görsel piksel koordinatları */
export interface StoredAnnotation {
  id: string
  shape: AnnotationShape
  label: string
  color: string
  origX: number
  origY: number
  origRadius: number
  origBbox?: [number, number, number, number]
  origPoints?: number[]
}

/** Canvas çizimi için (fit ölçeğine göre üretilir) */
export interface Annotation extends StoredAnnotation {
  x?: number
  y?: number
  radius?: number
  rect?: { x: number; y: number; width: number; height: number }
  points?: number[]
}

export interface ImageSession {
  id: string
  name: string
  /** img önizleme — Object URL (tercih) */
  previewUrl: string
  dataUrl: string
  base64: string
  imageWidth: number
  imageHeight: number
  annotations: StoredAnnotation[]
  detections: DetectedRegion[]
  results: SegmentResult[]
}

export interface SegmentResult {
  label: string
  predicted_class: string
  confidence: number
  crop_b64?: string
  mask_area?: number
}

export interface DetectedRegion {
  bbox: [number, number, number, number]
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

export interface HealthResponse {
  status: string
  sam_loaded: boolean
  vlm_loaded: boolean
  vlm_backend?: string
  device: string
  buffer_size: number
  known_classes: string[]
  pearl_lite?: Record<string, unknown>
}

export interface AnnotationInput {
  shape: AnnotationShape
  label: string
  origX: number
  origY: number
  origRadius: number
  origBbox?: [number, number, number, number]
  origPoints?: number[]
}
