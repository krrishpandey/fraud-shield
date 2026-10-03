const brl = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL', maximumFractionDigits: 2 })
const brl0 = new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL', maximumFractionDigits: 0 })
const int = new Intl.NumberFormat('en-US')

export const fmtBRL = (n: number | null | undefined, whole = false) =>
  n == null || Number.isNaN(n) ? 'n/a' : (whole ? brl0 : brl).format(n)
export const fmtInt = (n: number | null | undefined) => (n == null ? 'n/a' : int.format(n))
export const fmtPct = (p: number | null | undefined, digits = 0) =>
  p == null || Number.isNaN(p) ? 'n/a' : `${(p * 100).toFixed(digits)}%`
export const fmtMs = (n: number | null | undefined) => (n == null ? 'n/a' : `${n.toFixed(n < 10 ? 1 : 0)} ms`)
export const fmtDateTime = (iso: string) => {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return iso.replace('T', ' ').slice(0, 16)
}
export const shortHash = (h: string | null | undefined, n = 12) => (h ? (h.length > n ? `${h.slice(0, n)}...` : h) : 'n/a')
export const fmtNum = (v: number | string | null | undefined) => {
  if (v == null) return 'n/a'
  if (typeof v === 'string') return v
  if (Number.isInteger(v)) return int.format(v)
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)
}
