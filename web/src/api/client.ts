import type {
  AccountStory, AnalystRequest, AnalystResponse, AskRequest, AskResponse, AuditVerify, Booking, Counterfactual, DashboardMetrics,
  DecisionDetail, DecisionSummary, DemoBooking, ExplanationCheck, FirstScan, FirstScanDial, Health, LearningStatus, RetrainResponse, RollbackResponse,
  ScoreResponse, SimulateFeedbackResponse, StreamFeedRow, StreamFlagged, StreamMetricsResponse, StreamStatus,
} from './types'
import type { MonitorEstimate } from './types'

export const MOCK = import.meta.env.VITE_MOCK === '1' || import.meta.env.VITE_MOCK === 'true'
// Empty VITE_API_URL means same origin (the desktop app serves web/dist and the API together).
// `npm run dev` sets it to http://localhost:8080 through .env.development.
export const API_URL = ((import.meta.env.VITE_API_URL as string | undefined) ?? '').trim().replace(/\/$/, '')
export const API_LABEL = API_URL || 'same origin'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

type Fetcher = (path: string, init?: RequestInit) => Promise<Response>

let mockFetch: Fetcher | null = null
async function getFetcher(): Promise<Fetcher> {
  if (!MOCK) return (path, init) => fetch(API_URL + path, init)
  if (!mockFetch) {
    const mod = await import('../mocks/server')
    mockFetch = mod.mockFetch
  }
  return mockFetch
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const f = await getFetcher()
  let res: Response
  try {
    res = await f(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiError(0, `Cannot reach the tracd API (${API_LABEL}). Start the backend or run the console in mock mode.`)
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(body) })

export const api = {
  health: () => request<Health>('/health'),
  score: (b: Booking) => post<ScoreResponse>('/score', b),
  decisions: (params: { limit?: number; action?: string } = {}) => {
    const q = new URLSearchParams()
    q.set('limit', String(params.limit ?? 50))
    if (params.action) q.set('action', params.action)
    return request<DecisionSummary[]>(`/decisions?${q.toString()}`)
  },
  decision: (id: string) => request<DecisionDetail>(`/decisions/${encodeURIComponent(id)}`),
  analyst: (id: string, body: AnalystRequest) => post<AnalystResponse>(`/decisions/${encodeURIComponent(id)}/analyst`, body),
  ask: (id: string, body: AskRequest) => post<AskResponse>(`/decisions/${encodeURIComponent(id)}/ask`, body),
  firstScanDial: () => request<FirstScanDial>('/first-scan/dial'),
  setFirstScanDial: (level: string) => post<FirstScanDial>('/first-scan/dial', { level }),
  firstScan: (id: string, measured_weight_kg: number) =>
    post<FirstScan>(`/decisions/${encodeURIComponent(id)}/first-scan`, { measured_weight_kg }),
  accountStory: (id: string) => request<AccountStory>(`/decisions/${encodeURIComponent(id)}/account-story`),
  /** Analyst-only: smallest booker-controlled changes that would soften the decision (audited per view). */
  counterfactual: (id: string) => request<Counterfactual>(`/decisions/${encodeURIComponent(id)}/counterfactual`),
  checkExplanation: (id: string, text: string) =>
    post<ExplanationCheck>(`/decisions/${encodeURIComponent(id)}/explanation/check`, { text }),
  dashboard: (source: 'app' | 'stream' | 'all' = 'app') => request<DashboardMetrics>(`/dashboard/metrics?source=${source}`),
  auditVerify: () => request<AuditVerify>('/audit/verify'),
  learningStatus: () => request<LearningStatus>('/learning/status'),
  // advance_days: the realistic simulation only releases labels after their delays (analyst 1 day, disputes 7-60 days)
  simulateFeedback: (n: number, seed = 1, advanceDays = 90) =>
    post<SimulateFeedbackResponse>('/learning/simulate_feedback', { n, seed, advance_days: advanceDays }),
  retrain: (min_new_labels = 20) => post<RetrainResponse>('/learning/retrain', { min_new_labels }),
  rollback: (version: string) => post<RollbackResponse>('/learning/rollback', { version }),
  streamStart: (body: { rate: number; concurrency?: number; seed?: number }) => post<StreamStatus>('/stream/start', body),
  streamPause: () => post<StreamStatus>('/stream/pause', {}),
  streamResume: () => post<StreamStatus>('/stream/resume', {}),
  streamStop: () => post<StreamStatus>('/stream/stop', {}),
  streamStatus: () => request<StreamStatus>('/stream/status'),
  streamMetrics: () => request<StreamMetricsResponse>('/stream/metrics'),
  streamFlagged: (actions = 'hold,block') => request<StreamFlagged>(`/stream/flagged?actions=${actions}`),
  streamFeed: (limit = 30) => request<StreamFeedRow[]>(`/stream/feed?limit=${limit}`),
  monitorEstimate: () => request<MonitorEstimate>('/monitor/estimate'),
  /** GET /demo/bookings. In mock mode the local fixture is served by the mock layer. */
  demoBookings: async (): Promise<DemoBooking[]> => {
    const raw = await request<unknown[]>('/demo/bookings')
    return normalizeDemo(raw)
  },
}

/** Accepts either DemoBooking items or bare Booking objects (scenario taken from meta.scenario). */
export function normalizeDemo(raw: unknown[]): DemoBooking[] {
  return (raw ?? []).map((item, i) => {
    const it = item as Partial<DemoBooking> & Partial<Booking>
    if (it.booking) {
      return {
        scenario: it.scenario ?? String(it.booking.meta?.scenario ?? `scenario-${i + 1}`),
        title: it.title ?? it.booking.booking_id,
        description: it.description ?? '',
        expected: it.expected,
        booking: it.booking,
      }
    }
    const b = item as Booking
    const sc = String(b.meta?.scenario ?? b.booking_id ?? `scenario-${i + 1}`)
    return { scenario: sc, title: b.booking_id, description: '', booking: b }
  })
}
