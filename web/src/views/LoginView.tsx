import { useEffect, useRef, useState, type FormEvent } from 'react'

function Barcode({ seed }: { seed: string }) {
  let h = 0
  for (const c of seed) h = (h * 31 + c.charCodeAt(0)) >>> 0
  const bars: { w: number; gap: number }[] = []
  for (let i = 0; i < 64; i++) {
    h = (h * 1103515245 + 12345) >>> 0
    bars.push({ w: 1 + (h >>> 29), gap: 1 + ((h >>> 26) & 1) })
  }
  return (
    <div className="lg-barcode" aria-hidden="true">
      {bars.map((b, i) => (
        <i key={i} style={{ width: b.w, marginRight: b.gap }} />
      ))}
    </div>
  )
}

export default function LoginView({ onSignIn }: { onSignIn: (name: string) => void }) {
  const hostRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tipRef = useRef<HTMLDivElement>(null)
  const [glFailed, setGlFailed] = useState(false)
  const [name, setName] = useState('')
  const [code, setCode] = useState('')
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let disposed = false
    let dispose: (() => void) | null = null
    import('../lib/terminalScene')
      .then(({ createTerminalScene }) => {
        if (disposed || !hostRef.current || !canvasRef.current || !tipRef.current) return
        try {
          dispose = createTerminalScene(hostRef.current, canvasRef.current, tipRef.current).dispose
        } catch {
          setGlFailed(true)
        }
      })
      .catch(() => setGlFailed(true))
    return () => {
      disposed = true
      dispose?.()
    }
  }, [])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const n = name.trim()
    if (!n) return setErr('Enter your analyst name.')
    if (code.length < 4) return setErr('The pass code needs at least 4 characters.')
    onSignIn(n)
  }

  return (
    <div className="lg-root" data-testid="login-page">
      <div className="lg-scene" ref={hostRef}>
        <canvas ref={canvasRef} aria-label="3D sorting line. Drag to orbit, scroll to zoom, hover a parcel to read it." />
        <div className="lg-tip" ref={tipRef} />
        {glFailed && <div className="lg-nogl" />}
      </div>

      <form className="lg-label" onSubmit={submit} aria-labelledby="lg-title" noValidate>
        <div className="lg-row lg-from">
          <div className="lg-brand">Tracd</div>
          <div className="lg-svc" aria-hidden="true">
            A
          </div>
        </div>

        <div className="lg-row">
          <h1 id="lg-title" className="lg-big">
            Sign in
          </h1>
        </div>

        <div className="lg-row lg-fields">
          <label className="lg-field">
            <span className="lg-k">Analyst name</span>
            <input
              autoFocus
              autoComplete="username"
              value={name}
              onChange={(e) => {
                setName(e.target.value)
                setErr(null)
              }}
              placeholder="e.g. Krrish"
              data-testid="login-name"
            />
          </label>
          <label className="lg-field">
            <span className="lg-k">Pass code</span>
            <input
              type="password"
              autoComplete="current-password"
              value={code}
              onChange={(e) => {
                setCode(e.target.value)
                setErr(null)
              }}
              placeholder="••••"
              data-testid="login-code"
            />
          </label>
          {err && (
            <p className="lg-err" role="alert" data-testid="login-error">
              {err}
            </p>
          )}
          <button type="submit" className="lg-submit" data-testid="login-submit">
            Open the console
            <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2.4" />
            </svg>
          </button>
          {/* Honest label: this gate only names who is at the desk; it is not access control (lib/session.ts). */}
          <p className="lg-k" data-testid="login-demo-note">
            Demo sign-in: any name and a code of 4 or more characters. It labels who is at the desk; the API has no accounts.
          </p>
        </div>

        <div className="lg-row lg-code">
          <Barcode seed="tracd" />
        </div>

        <div className={`lg-stamp${name.trim() && code.length >= 4 ? ' is-ok' : ''}`} aria-hidden="true">
          {name.trim() && code.length >= 4 ? 'Allow' : 'Hold'}
        </div>
      </form>

    </div>
  )
}
