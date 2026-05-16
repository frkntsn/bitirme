import { useState, useEffect, useCallback } from 'react'
import { fetchHealth } from '../api/client'
import type { HealthResponse } from '../types'

export function useBackendHealth(pollMs = 15000) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [online, setOnline] = useState<boolean | null>(null)

  const refresh = useCallback(async () => {
    try {
      const h = await fetchHealth()
      setHealth(h)
      setOnline(true)
    } catch {
      setHealth(null)
      setOnline(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, pollMs)
    return () => clearInterval(id)
  }, [refresh, pollMs])

  return { health, online, refresh }
}
