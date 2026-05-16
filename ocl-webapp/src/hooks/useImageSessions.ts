import { useState, useCallback } from 'react'
import { ImageSession, StoredAnnotation, DetectedRegion, SegmentResult } from '../types'

function newId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function revokePreview(session: ImageSession) {
  if (session.previewUrl.startsWith('blob:')) {
    try {
      URL.revokeObjectURL(session.previewUrl)
    } catch {
      /* ignore */
    }
  }
}

export function useImageSessions() {
  const [sessions, setSessions] = useState<ImageSession[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)

  const active = sessions.find((s) => s.id === activeId) ?? null

  const updateSession = useCallback(
    (id: string, patch: Partial<Pick<ImageSession, 'annotations' | 'detections' | 'results'>>) => {
      setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)))
    },
    []
  )

  const addFromFile = useCallback(
    (file: File, dataUrl: string, base64: string, imageWidth: number, imageHeight: number) => {
      const previewUrl = URL.createObjectURL(file)
      const session: ImageSession = {
        id: newId(),
        name: file.name || 'görsel',
        previewUrl,
        dataUrl,
        base64,
        imageWidth,
        imageHeight,
        annotations: [],
        detections: [],
        results: [],
      }
      setSessions((prev) => [...prev, session])
      setActiveId(session.id)
      return session.id
    },
    []
  )

  const selectSession = useCallback((id: string) => {
    setActiveId(id)
  }, [])

  const removeSession = useCallback((id: string) => {
    setSessions((prev) => {
      const removed = prev.find((s) => s.id === id)
      if (removed) revokePreview(removed)
      const next = prev.filter((s) => s.id !== id)
      setActiveId((cur) => (cur === id ? next[next.length - 1]?.id ?? null : cur))
      return next
    })
  }, [])

  const removeActive = useCallback(() => {
    if (activeId) removeSession(activeId)
  }, [activeId, removeSession])

  const patchActive = useCallback(
    (patch: {
      annotations?: StoredAnnotation[]
      detections?: DetectedRegion[]
      results?: SegmentResult[]
    }) => {
      if (!activeId) return
      updateSession(activeId, patch)
    },
    [activeId, updateSession]
  )

  const clearAllSessions = useCallback(() => {
    setSessions((prev) => {
      prev.forEach(revokePreview)
      return []
    })
    setActiveId(null)
  }, [])

  return {
    sessions,
    active,
    activeId,
    addFromFile,
    selectSession,
    removeSession,
    removeActive,
    patchActive,
    updateSession,
    setSessions,
    clearAllSessions,
  }
}
