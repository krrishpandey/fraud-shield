import type { ReactNode } from 'react'

export function ErrorBox({ message, onRetry, testId }: { message: string; onRetry?: () => void; testId?: string }) {
  return (
    <div role="alert" data-testid={testId ?? 'error-box'} className="panel border-danger p-4" style={{ borderColor: 'var(--danger)' }}>
      <p className="font-semibold text-danger">Request failed</p>
      <p className="mt-1 text-ink-2">{message}</p>
      {onRetry && (
        <button type="button" className="btn mt-3" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}

export function Loading({ what }: { what: string }) {
  return (
    <p className="text-muted" role="status" aria-live="polite">
      Loading {what}...
    </p>
  )
}

export function Section({
  title,
  aside,
  children,
  id,
  testId,
}: {
  title: string
  aside?: ReactNode
  children: ReactNode
  id?: string
  testId?: string
}) {
  const hid = id ? `${id}-h` : undefined
  return (
    <section className="panel p-4" aria-labelledby={hid} id={id} data-testid={testId}>
      <div className="mb-2.5 flex flex-wrap items-baseline justify-between gap-2">
        <h2 id={hid} className="text-[0.95rem] font-bold tracking-tight">
          {title}
        </h2>
        {aside && <div className="text-sm text-muted">{aside}</div>}
      </div>
      {children}
    </section>
  )
}

export function PageTitle({ title, sub }: { title: string; sub?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <h1 className="condensed text-[1.45rem] leading-tight font-extrabold tracking-tight">{title}</h1>
      {sub && <p className="max-w-[90ch] text-[0.85rem] text-ink-2">{sub}</p>}
    </div>
  )
}
