import { useEffect, useState } from 'react'

/** Resolves CSS custom properties to concrete colors (for SVG chart libraries). Re-reads on theme change. */
export function useCssVars(names: string[]): Record<string, string> {
  const key = names.join('|')
  const read = () => {
    const cs = getComputedStyle(document.documentElement)
    const out: Record<string, string> = {}
    for (const n of key.split('|')) out[n] = cs.getPropertyValue(n).trim() || '#888'
    return out
  }
  const [vals, setVals] = useState<Record<string, string>>(read)
  useEffect(() => {
    const cs = getComputedStyle(document.documentElement)
    const update = () => {
      const out: Record<string, string> = {}
      for (const n of key.split('|')) out[n] = cs.getPropertyValue(n).trim() || '#888'
      setVals(out)
    }
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    mq.addEventListener('change', update)
    const mo = new MutationObserver(update)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => {
      mq.removeEventListener('change', update)
      mo.disconnect()
    }
  }, [key])
  return vals
}
