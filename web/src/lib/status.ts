import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api } from '../api/client'
import type { DecisionSummary, Health } from '../api/types'

/* Console-wide status read from existing endpoints: /health, /dashboard/metrics, /decisions. Read-only. */
export interface ConsoleStatus {
  health: Health | null
  healthError: string | null
  latencyMs: number | null
  openCases: DecisionSummary[]
  recent: DecisionSummary[]
  refresh: () => void
}

export const StatusContext = createContext<ConsoleStatus>({
  health: null,
  healthError: null,
  latencyMs: null,
  openCases: [],
  recent: [],
  refresh: () => {},
})
export const useStatus = () => useContext(StatusContext)

const EVERY_MS = 15000

export function useConsoleStatus(): ConsoleStatus {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [latencyMs, setLatency] = useState<number | null>(null)
  const [recent, setRecent] = useState<DecisionSummary[]>([])
  const [tick, setTick] = useState(0)
  const refresh = useCallback(() => setTick((t) => t + 1), [])

  useEffect(() => {
    let alive = true
    const load = async () => {
      api
        .health()
        .then((h) => {
          if (!alive) return
          setHealth(h)
          setHealthError(null)
        })
        .catch((e) => alive && setHealthError(e instanceof Error ? e.message : String(e)))
      api
        .dashboard('all')
        .then((m) => alive && setLatency(m.totals.bookings ? m.latency.p50_ms : null))
        .catch(() => {})
      api
        .decisions({ limit: 200 })
        .then((d) => alive && setRecent(d))
        .catch(() => {})
    }
    load()
    const t = setInterval(load, EVERY_MS)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [tick])

  const openCases = recent.filter((r) => r.action !== 'allow' && r.analyst_label == null)
  return { health, healthError, latencyMs, openCases, recent, refresh }
}

const LAST_KEY = 'fs-last-decision'
export function rememberDecision(id: string) {
  try {
    localStorage.setItem(LAST_KEY, id)
  } catch {
    /* ignore */
  }
}
export function lastDecision(): string | null {
  try {
    return localStorage.getItem(LAST_KEY)
  } catch {
    return null
  }
}
