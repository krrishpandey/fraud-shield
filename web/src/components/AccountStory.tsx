import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { AccountStory as Story, Action } from '../api/types'

/*
  The product's core idea on one screen: new receivers are everyday business for a real seller,
  new senders under the same account are not. Counts use the same definitions as the model features,
  so they match the explanation word for word.
*/

function useStory(id: string) {
  const [story, setStory] = useState<Story | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    api
      .accountStory(id)
      .then((s) => alive && setStory(s))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [id])
  return { story, error }
}

const per10 = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(v < 1 ? 2 : 1))

function Squares({ size, filled, hot, label }: { size: number; filled: number; hot: boolean; label: string }) {
  return (
    <span className={`story-squares${hot ? ' is-hot' : ''}`} role="img" aria-label={label}>
      {Array.from({ length: size }, (_, i) => (
        <span key={i} className={i < filled ? 'sq on' : 'sq'} />
      ))}
    </span>
  )
}

function CompareRow({
  title,
  count,
  size,
  usual,
  hot,
  testId,
}: {
  title: string
  count: number
  size: number
  usual: number | null
  hot: boolean
  testId: string
}) {
  return (
    <div className="story-row" data-testid={testId} data-count={count} data-usual={usual ?? ''}>
      <div className="story-row-title">{title}</div>
      <Squares size={size} filled={count} hot={hot} label={`${count} of ${size}`} />
      <div className="story-row-count tnum">
        <strong>{count}</strong> of {size}
      </div>
      <div className="story-row-usual tnum">{usual == null ? 'no earlier bookings to compare' : `usually ${per10(usual)}`}</div>
    </div>
  )
}

function Strip({ story }: { story: Story }) {
  const bs = story.bookings
  const step = 10
  const w = bs.length * step
  const others = bs.filter((b) => !b.own_goods).length
  return (
    <div className="story-strip">
      <div className="story-lanes" aria-hidden="true">
        <span>Its own goods</span>
        <span>Someone else's goods</span>
      </div>
      <div className="story-strip-plot">
        <svg
          viewBox={`0 0 ${w} 64`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`Last ${bs.length} bookings: ${bs.length - others} the account's own goods, ${others} paid for other senders`}
        >
          <line x1={0} x2={w} y1={32} y2={32} className="story-mid" />
          {bs.map((b, i) => {
            const x = i * step + 2
            const cls = b.current ? 'bar current' : b.own_goods ? 'bar own' : b.new_sender ? 'bar new' : 'bar other'
            return (
              <rect key={b.booking_id} x={x} width={step - 4} y={b.own_goods ? 3 : 35} height={26} className={cls}>
                <title>{`${b.booked_at.replace('T', ' ').slice(0, 16)}, ${b.origin} to ${b.dest}${b.own_goods ? ', own goods' : b.new_sender ? ', first time paying for this sender' : ', paid for another sender'}`}</title>
              </rect>
            )
          })}
        </svg>
        <div className="story-dates tnum" aria-hidden="true">
          <span>{bs[0]?.booked_at.slice(0, 10)}</span>
          <span>this booking</span>
        </div>
      </div>
    </div>
  )
}

export function AccountStory({ decisionId, action }: { decisionId: string; action: Action }) {
  const { story, error } = useStory(decisionId)
  if (error) {
    return (
      <p className="text-[0.85rem] text-muted" data-testid="story-unavailable">
        Account history is not shown here: {error}
      </p>
    )
  }
  if (!story) return <p className="text-muted">Loading the account's bookings...</p>

  const { last10, usual } = story
  const window = story.bookings.slice(0, -1).slice(-last10.size)
  const sendersHot = usual != null && last10.new_senders >= usual.new_senders_per10 + 3
  const receiversUsual = usual != null && Math.abs(last10.new_receivers - usual.new_receivers_per10) <= 2
  const allOwn = window.length > 0 && window.every((b) => b.own_goods)
  let insight: string | null = null
  if (sendersHot && receiversUsual) insight = 'New receivers are everyday business for this account. New senders are not.'
  else if (allOwn) insight = 'Every one of these bookings was the account shipping its own goods.'

  return (
    <div data-testid="account-story" style={{ ['--act' as string]: `var(--act-${action})` }}>
      {last10.size > 0 ? (
        <>
          <CompareRow
            title="Shipped to a receiver for the first time"
            count={last10.new_receivers}
            size={last10.size}
            usual={usual?.new_receivers_per10 ?? null}
            hot={false}
            testId="story-receivers"
          />
          <CompareRow
            title="Paid for a sender for the first time"
            count={last10.new_senders}
            size={last10.size}
            usual={usual?.new_senders_per10 ?? null}
            hot={sendersHot}
            testId="story-senders"
          />
          {insight && (
            <p className="story-insight" data-testid="story-insight">
              {insight}
            </p>
          )}
          <Strip story={story} />
        </>
      ) : (
        <p className="text-ink-2">This is the account's first booking, so there is nothing to compare it with yet.</p>
      )}
    </div>
  )
}
