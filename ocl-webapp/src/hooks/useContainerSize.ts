import { useEffect, useState, RefObject } from 'react'

export function useContainerSize(ref: RefObject<HTMLElement | null>) {
  const [size, setSize] = useState({ width: 700, height: 480 })

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const update = () => {
      setSize({ width: el.clientWidth, height: el.clientHeight })
    }
    update()

    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])

  return size
}
