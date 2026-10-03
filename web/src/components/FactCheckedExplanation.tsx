import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { DecisionDetail, ExplanationCheck, NumberSpan } from '../api/types'

/*
  The language model's explanation, with every number marked as checked against the decision record.
  "Try a wrong number" runs the same validator on edited text, so anyone can watch it refuse an
  explanation that invents a figure.
*/

function Marked({ text, numbers }: { text: string; numbers: NumberSpan[] }) {
  const parts: React.ReactNode[] = []
  let at = 0
  numbers.forEach((n, i) => {
    if (n.start > at) parts.push(text.slice(at, n.start))
    parts.push(
      <span
        key={i}
        className={n.ok ? 'num-ok' : 'num-bad'}
        title={n.ok ? 'Found in the decision record' : 'Not in the decision record'}
        data-testid={n.ok ? undefined : 'num-bad'}
      >
        {text.slice(n.start, n.end)}
      </span>,
    )
    at = n.end
  })
  parts.push(text.slice(at))
  return <>{parts}</>
}

function useCheck(id: string, text: string | null) {
  const [check, setCheck] = useState<ExplanationCheck | null>(null)
  useEffect(() => {
    if (!text) return
    let alive = true
    api
      .checkExplanation(id, text)
      .then((c) => alive && setCheck(c))
      .catch(() => alive && setCheck(null))
    return () => {
      alive = false
    }
  }, [id, text])
  return check
}

function plainProblem(p: string): string {
  if (p.startsWith('number not in record: ')) return `${p.slice(22)} is not in the decision record`
  if (p.startsWith('id or code not in record: ')) return `${p.slice(26)} is not an id from this booking`
  if (p.startsWith('does not name the action')) return 'It does not say what the system decided'
  if (p.startsWith('mentions ')) return 'It leaves out the main reasons'
  if (p.startsWith('banned phrase: ')) return `It claims too much (${p.slice(15)})`
  if (p.startsWith('too long')) return 'It is too long'
  return p
}

function TryIt({ d, original }: { d: DecisionDetail; original: string }) {
  const [draft, setDraft] = useState(original)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<{ text: string; check: ExplanationCheck } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const run = async () => {
    setBusy(true)
    setErr(null)
    try {
      setResult({ text: draft, check: await api.checkExplanation(d.decision_id, draft) })
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="try-box">
      <label htmlFor="try-input" className="block text-[0.82rem] text-ink-2">
        Change a number, or add a claim, then check it the way every explanation is checked.
      </label>
      <textarea
        id="try-input"
        data-testid="explanation-try-input"
        className="field mt-1 min-h-[7rem] resize-y leading-relaxed"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
      />
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" className="btn btn-primary" onClick={run} disabled={busy || !draft.trim()} data-testid="explanation-try-check">
          {busy ? 'Checking...' : 'Check this text'}
        </button>
        <button type="button" className="btn" onClick={() => { setDraft(original); setResult(null) }}>
          Reset
        </button>
      </div>
      {err && <p className="mt-2 text-[0.85rem] text-danger" role="alert">{err}</p>}
      {result && (
        <div className="try-result" data-testid="explanation-try-result" data-ok={result.check.ok} role="status">
          <p className="try-verdict">
            {result.check.ok ? 'Passes. Analysts would see this text.' : 'Rejected. Analysts would see the fixed template instead.'}
          </p>
          {result.check.problems.length > 0 && (
            <ul className="mt-1 list-disc pl-5 text-[0.85rem]">
              {result.check.problems.map((p) => (
                <li key={p}>{plainProblem(p)}</li>
              ))}
            </ul>
          )}
          <p className="mt-2 leading-relaxed">
            <Marked text={result.text} numbers={result.check.numbers} />
          </p>
        </div>
      )}
    </div>
  )
}

export function FactCheckedExplanation({ d, polling }: { d: DecisionDetail; polling: boolean }) {
  const e = d.explanation
  const check = useCheck(d.decision_id, e?.text ?? null)
  const [trying, setTrying] = useState(false)

  if (!e) {
    const failed = d.explanation_status === 'failed'
    return (
      <p className="text-ink-2" data-testid="explanation-status" data-status={d.explanation_status} role="status" aria-live="polite">
        {failed
          ? 'No explanation was produced for this decision.'
          : polling
            ? 'Writing the explanation. It is checked against the decision record before it is shown.'
            : 'The explanation is still being written. Reload to check again.'}
      </p>
    )
  }
  const found = check?.numbers.filter((n) => n.ok).length ?? 0
  const total = check?.numbers.length ?? 0
  return (
    <div data-testid="explanation" aria-live="polite">
      <p className="explain-text" data-testid="explanation-text">
        {check ? <Marked text={e.text} numbers={check.numbers} /> : e.text}
      </p>
      <p className="explain-meta">
        <span data-testid="explanation-source" data-source={e.source}>
          {e.source === 'llm' ? `Written by a language model (${e.model_id ?? 'unknown'}).` : 'Fixed template built from the decision record.'}
        </span>{' '}
        <span data-testid="explanation-validator" data-valid={e.valid}>
          {!e.valid
            ? 'The model text failed the check, so the template is shown.'
            : check && total > 0
              ? `Checked against the decision record: ${found} of ${total} numbers found.`
              : 'Checked against the decision record.'}
        </span>
      </p>
      {check && (
        <button type="button" className="link-btn" aria-expanded={trying} onClick={() => setTrying((t) => !t)} data-testid="explanation-try-toggle">
          {trying ? 'Close the test' : 'Try slipping in a wrong number'}
        </button>
      )}
      {trying && <TryIt d={d} original={e.text} />}
    </div>
  )
}
