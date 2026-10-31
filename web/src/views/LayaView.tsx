import { api } from '../api/client'
import { ErrorBox, Loading, PageTitle } from '../components/common'
import { useAsync } from '../lib/useAsync'
import '../verify.css'

const QUESTION_TEXT: Record<string, string> = {
  misuse: 'Is someone other than the owner using this account?',
  foreign_senders: 'Is the account paying for senders it never used before?',
  payoff_max: 'Is this parcel much more expensive than usual?',
  drop_consignee: 'Does the receiver look like a reshipping drop?',
  risk_level: 'How risky is it (5 levels)?',
  action: 'Which action should the carrier take?',
}

const SYSTEM_TEXT: Record<string, string> = {
  'B2 LightGBM (current decider)': 'LightGBM (decides today)',
  'S0 3-input stack (control)': 'LightGBM + small stack (control)',
  'M1 Laya v1': 'Laya v1 (fine-tuned)',
  'M2 Laya v2 decider': 'Laya v2 (fine-tuned, with LightGBM as a hint)',
  'M2-solo (LightGBM line withheld)': 'Laya v2 without the LightGBM hint',
}

/** Laya: what it is, whether it runs on this computer, and how its fine-tuning went (from the repo's result files). */
export default function LayaView() {
  const { data, error, loading, reload } = useAsync(() => api.layaStatus(), [])
  const best = data?.pr_auc ? Math.max(...Object.values(data.pr_auc)) : 1
  return (
    <div>
      <PageTitle title="Laya model" />
      <div className="page how">
        <p className="vf-intro">
          Laya is a language model that was <b>fine-tuned</b> (extra-trained) on our bookings to answer yes/no fraud questions in words. In
          Tracd it explains <b>what looks wrong</b> with a held booking. LightGBM still makes the decision, because it scored better in the test below.
        </p>
        {error && <ErrorBox message={error} onRetry={reload} />}
        {loading && <Loading what="Laya's status" />}
        {data && (
          <>
            <section className="panel">
              <div className="card-h">
                <h2>The fine-tuning</h2>
              </div>
              <div className="card-b">
                <p className="vf-note">{data.model}.</p>
                <ol className="how-steps">
                  {data.runs.map((r, i) => (
                    <li key={r.name} className="is-model">
                      <span className="how-n mono">{i + 1}</span>
                      <div className="min-w-0">
                        <div className="how-t">
                          <b>{r.name}</b>
                          <span className="how-who">{r.where}</span>
                        </div>
                        <p>Trained on {r.data || 'see the result file'}.</p>
                        {r.questions && (
                          <ul className="vf-reasons">
                            {r.questions.map((q) => (
                              <li key={q}>{QUESTION_TEXT[q] ?? q}</li>
                            ))}
                          </ul>
                        )}
                        {r.trainable_params != null && <p>{(r.trainable_params / 1e6).toFixed(1)} million weights were trained; the rest stayed frozen.</p>}
                        <code className="how-where">{r.result_file}</code>
                      </div>
                    </li>
                  ))}
                </ol>
              </div>
            </section>

            <section className="panel">
              <div className="card-h">
                <h2>The test: who finds fraud better?</h2>
              </div>
              <div className="card-b">
                <p className="vf-note">
                  Score on bookings none of the models saw while training (PR-AUC: 1.0 would be perfect; higher is better). Mean of 10 runs.
                </p>
                <ul className="laya-bars" data-testid="laya-scores">
                  {Object.entries(data.pr_auc ?? {}).map(([k, v]) => (
                    <li key={k}>
                      <span>{SYSTEM_TEXT[k] ?? k}</span>
                      <span className="laya-bar" aria-hidden="true">
                        <span style={{ width: `${(v / best) * 100}%` }} />
                      </span>
                      <b className="tnum">{v.toFixed(3)}</b>
                    </li>
                  ))}
                </ul>
                <p className="vf-big mt-3">
                  Result: <b>{(data.verdict ?? '').replace(/\s*\(.*\)$/, '')}</b>. Laya was only allowed to decide if it matched LightGBM; it did not, so it explains instead.
                </p>
              </div>
            </section>

            <section className="panel">
              <div className="card-h">
                <h2>Files</h2>
              </div>
              <div className="card-b">
                <ul className="vf-reasons">
                  {data.files.map((f) => (
                    <li key={f}>
                      <code className="how-where">{f}</code>
                    </li>
                  ))}
                </ul>
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  )
}
