import { Link } from 'react-router-dom'
import { PageTitle } from '../components/common'
import '../verify.css'

type Step = { n: string; title: string; what: string; where: string; who: string; tone?: 'model' | 'voice' | 'off' }

// One booking, start to finish. Every file named here exists in the repo; docs/WORKFLOW.md has the same steps in more depth.
const LIVE: Step[] = [
  { n: '1', title: 'A booking arrives', who: 'API', where: 'fraudshield/api/app.py  POST /score',
    what: 'A shipper books a parcel: account, sender, from, to, weight, size, service, price, login device. Nothing is printed yet.' },
  { n: '2', title: 'Look up the account history', who: 'Feature store', where: 'fraudshield/features/featurize.py',
    what: 'About 70 numbers (71 in the model) are computed from the account\'s own past bookings, only from BEFORE this booking: how many new senders it pays for, cost compared with its usual cost, weight compared with its usual weight, how old the login device is, booking hour, how many other accounts ship to the same receiver.' },
  { n: '3', title: 'LightGBM gives a fraud score', who: 'LightGBM (decides)', where: 'fraudshield/models/gbm.py  +  artifacts/models/B2_F.lgb', tone: 'model',
    what: 'The trained LightGBM model turns those numbers into a probability of fraud between 0 and 1 (calibrated with Platt scaling, so 0.9 means about 9 in 10). It takes a few milliseconds. This is the score you see as "Risk".' },
  { n: '4', title: 'Laya says what looks wrong', who: 'Laya (explains)', where: 'fraudshield/voice/laya_explain.py  +  Laya model page', tone: 'model',
    what: 'Laya is a language model fine-tuned on our bookings. For a held booking it answers four yes/no questions (someone else using the account? new senders? unusually expensive? a reshipping drop?) and the app writes the yes-answers as plain sentences. It does not decide: on the held-out test it found fraud worse than LightGBM (PR-AUC 0.356 vs 0.777). When Laya is not connected (no LAYA_API_KEY, weights not on this computer), the same box shows clearly labelled demo text built from the booking\'s own facts.' },
  { n: '5', title: 'Rules add hard facts', who: 'Rules', where: 'fraudshield/api/pipeline.py  default_rules()',
    what: 'A receiver that looks like a reshipping drop raises the risk. A link to confirmed fraud, or an account the owner already reported, is a "hard signal": only hard signals may block.' },
  { n: '6', title: 'The cost rule picks the action', who: 'Cost policy', where: 'fraudshield/policy/decide.py  +  config/costs.yaml',
    what: 'For each of 6 actions (allow, allow with a check at first scan, ask the owner, analyst review, hold, block) it adds up the expected cost: money lost if it is fraud and we let it through, versus annoying an honest customer, versus analyst time. The cheapest allowed action wins.' },
  { n: '7', title: 'Written to the audit log', who: 'Audit log', where: 'fraudshield/audit/log.py  ->  artifacts/audit/audit.jsonl',
    what: 'The decision is appended with a SHA-256 hash that includes the previous record\'s hash. Changing any old record breaks the chain, and the Audit page shows it.' },
]

const VERIFY: Step[] = [
  { n: '8', title: 'Held? Check the phone number first', who: 'Tracd Verify', where: 'fraudshield/voice/service.py  start()  +  config/owners.yaml', tone: 'voice',
    what: 'Only for "hold" or "ask the owner". If the owner\'s number changed in the last 30 days, Tracd does not call (a fraudster could have changed it) and the booking goes to an analyst.' },
  { n: '9', title: 'Bolna calls the owner', who: 'Bolna voice agent', where: 'fraudshield/voice/bolna.py  POST https://api.bolna.ai/call', tone: 'voice',
    what: 'The agent speaks English, says it will never ask for an OTP, password or card number, reads out the booking and asks "Was this you?". If yes, it asks the pickup address and whether anything changed. Without a Bolna key, a clearly labelled simulated call runs the same steps (fraudshield/voice/script.py).' },
  { n: '10', title: 'The answer comes back', who: 'Webhook', where: 'fraudshield/voice/router.py  POST /voice/webhook  +  voice/outcome.py', tone: 'voice',
    what: 'Bolna posts the call result: status, transcript and the extracted answers (owner_confirmed, detail_answer, change_reason, language). The same message twice only counts once.' },
  { n: '11', title: 'Decide again with the answer', who: 'LightGBM again + cost rule', where: 'fraudshield/api/pipeline.py  rescore_after_owner()  +  voice/damping.py', tone: 'model',
    what: 'NO: block the booking, lock the account, open an analyst case. YES with a matching pickup address: the change the owner explained (new warehouse, new staff login) is treated as normal, LightGBM scores again, the "yes" lowers the risk further, and the cost rule runs again. Usually it is released with a check at first scan. A hard signal is never released by a yes. UNSURE, wrong address or no answer (after one retry and an SMS): an analyst decides. Nothing is ever released automatically.' },
  { n: '12', title: 'Logged as new records', who: 'Audit log', where: 'artifacts/audit/audit.jsonl',
    what: 'Every call step is a NEW record that points to the original decision id. The decision itself is never edited.' },
]

const OFFLINE: Step[] = [
  { n: 'A', title: 'Build the data', who: 'Data pipeline', where: 'scripts/build_dataset.py',
    what: 'Real Olist bookings (about 100,000 from 3,095 sellers) plus an invented layer of accounts, logins and devices, plus about 1% injected fraud in 7 patterns. Split by time: train, calibration, then a test window (15 May to 31 Aug 2018) the model never sees.' },
  { n: 'B', title: 'Train LightGBM', who: 'LightGBM', where: 'scripts/train_gbm.py  ->  artifacts/models/B2_F.lgb', tone: 'model',
    what: 'Trained on the train window, calibrated on the calibration window, tested on the test window. Measured: PR-AUC 0.777 on the test window, 10 injection seeds (artifacts/results_gbm.md).' },
  { n: 'C', title: 'Fine-tune Laya (lost)', who: 'Laya', where: 'scripts/laya_train.py (Kaggle GPU)  ->  artifacts/results_laya_v2.md', tone: 'off',
    what: 'Fine-tuned to answer the fraud questions. It had to beat LightGBM on fraud types it never saw; it did not, so LightGBM keeps deciding.' },
  { n: 'D', title: 'Keep learning', who: 'Learning loop', where: 'fraudshield/learning/',
    what: 'Analyst labels and owner "no" answers become new labels. A retrain is only deployed if it passes a pre-registered quality gate (Learning page).' },
]

function Steps({ steps }: { steps: Step[] }) {
  return (
    <ol className="how-steps">
      {steps.map((s) => (
        <li key={s.n} className={s.tone ? `is-${s.tone}` : ''}>
          <span className="how-n mono">{s.n}</span>
          <div className="min-w-0">
            <div className="how-t">
              <b>{s.title}</b>
              <span className="how-who">{s.who}</span>
            </div>
            <p>{s.what}</p>
            <code className="how-where">{s.where}</code>
          </div>
        </li>
      ))}
    </ol>
  )
}

export default function HowView() {
  return (
    <div>
      <PageTitle title="How it works" />
      <div className="page how">
        <p className="vf-intro">
          What happens to one booking, step by step, and which file does each step. Colour key: <span className="how-key is-model">model</span>{' '}
          <span className="how-key is-voice">voice call</span> <span className="how-key is-off">switched off in this build</span>
        </p>
        <section className="panel">
          <div className="card-h">
            <h2>When a booking is made (under 300 ms)</h2>
          </div>
          <div className="card-b">
            <Steps steps={LIVE} />
          </div>
        </section>
        <section className="panel">
          <div className="card-h">
            <h2>Tracd Verify: when a booking is held (about 1 to 2 minutes)</h2>
          </div>
          <div className="card-b">
            <Steps steps={VERIFY} />
            <p className="vf-note">
              See it on the <Link className="row-link" to="/calls">Owner calls</Link> page: pick a call and read the four numbered steps.
            </p>
          </div>
        </section>
        <section className="panel">
          <div className="card-h">
            <h2>Done before the app runs (training)</h2>
          </div>
          <div className="card-b">
            <Steps steps={OFFLINE} />
          </div>
        </section>
        <section className="panel">
          <div className="card-h">
            <h2>Real, invented, simulated</h2>
          </div>
          <div className="card-b">
            <ul className="vf-reasons">
              <li><b>Real:</b> Olist bookings (dates, places, weights, prices), the trained LightGBM, every score and cost on screen.</li>
              <li><b>Invented:</b> account logins, devices and billing (no public data has them), and the injected fraud rows.</li>
              <li><b>Simulated (labelled in yellow):</b> owner phone calls when there is no Bolna key or no real number; the owner's answer then comes from the dataset label or the demo script.</li>
              <li><b>Demo labels:</b> Indian city and company names on the three demo scenarios; the data underneath is Olist.</li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  )
}
