import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { useAsync } from '../lib/useAsync'
import '../verify.css'

/** Plain-English reasons for this booking on the Decision page (fraudshield/voice/laya_explain.py). Labelled "Laya"
 * only when the fine-tuned Laya answered; otherwise they are read from the booking's own facts. */
export function LayaWhy({ id }: { id: string }) {
  const { data, error } = useAsync(() => api.decisionLaya(id), [id])
  if (error || !data) return null
  const wrong = data.items.filter((i) => i.yes)
  const v = data.verification
  return (
    <div className="laya-why" data-testid="laya-why" data-source={data.source}>
      {v && (
        <p className="laya-call">
          This booking was held and the account owner was called ("Was this you?").{' '}
          <Link className="row-link" to={`/calls?d=${encodeURIComponent(id)}`}>
            See the call and transcript
          </Link>
        </p>
      )}
      <p className="laya-h">{data.source === 'demo' ? 'What looks wrong' : 'Laya: what looks wrong'}</p>
      {wrong.length ? (
        <ul>
          {wrong.map((i) => (
            <li key={i.id}>
              {i.text}
              {i.probability != null && <span className="text-muted"> ({Math.round(i.probability * 100)}% yes)</span>}
            </li>
          ))}
        </ul>
      ) : (
        <p>Nothing unusual.</p>
      )}
    </div>
  )
}
