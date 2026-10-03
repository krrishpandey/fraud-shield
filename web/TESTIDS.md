# FraudShield console: test ids for E2E

All selectors are `[data-testid="..."]`. Routing is hash based: `/#/`, `/#/decisions/<id>`, `/#/queue`, `/#/dashboard`, `/#/learning`, `/#/audit`.

Run modes:
- Mock (no backend): `npm run dev:mock` (loads `.env.mock`, `VITE_MOCK=1`). In-memory fixtures, resets on reload. Seeded decisions `dec_000101` to `dec_000107` exist at start; new scores get `dec_000108` onward. Explanations stay pending for 1.5 s after scoring, then appear.
- Real: start the API on 8080, then `npm run dev` (`.env.development` sets `VITE_API_URL=http://localhost:8080`). A production build with empty `VITE_API_URL` calls the API on the same origin.

## Shell (every page)
| id | element | notes |
|---|---|---|
| `health-status` | API status in the top bar | `data-state` = `ok` / `degraded` / `down` / `unknown`; `data-laya-mode` = `local` / `http` / `cached`; `data-gpu` = `true` / `false` |
| `mock-banner` | yellow "Mock data" banner | present only in mock mode |
| `nav-score`, `nav-queue`, `nav-dashboard`, `nav-learning`, `nav-audit` | sidebar links | |

## Score a booking (`/#/`)
| id | element | notes |
|---|---|---|
| `demo-booking-<scenario>` | demo booking row (click selects, double-click scores) | scenarios in the fixture: `legit-tenured`, `hard-negative-new-state`, `takeover`, `reshipping-drop`, `weight-manipulation`. With the real API the scenario key comes from GET /demo/bookings |
| `selected-booking` | details panel of the selected booking | |
| `score-button` | scores the selected booking (POST /score) then navigates to the decision | |
| `score-error` | error box when scoring fails | |
| `manual-toggle` | shows or hides the manual form | |
| `manual-form` | manual booking form | |
| `manual-<field>` | text and number inputs, e.g. `manual-account_id`, `manual-carrier_cost` | selects use label text |
| `manual-score-button` | scores the manual booking | |

## Decision detail (`/#/decisions/<decision_id>`)
| id | element | notes |
|---|---|---|
| `decision-detail` | page root | `data-decision-id` |
| `decision-error` | load error | |
| `action-badge` | routing-label stamp with action text and meaning | `data-action` = action key |
| `degraded-flag`, `explored-flag` | shown only when flagged | |
| `explanation-panel` | "Why" section | |
| `explanation-status` | shown while pending | `data-status` |
| `explanation` | wrapper once ready | |
| `explanation-text` | explanation text | |
| `explanation-source` | source chip | `data-source` = `llm` / `template` |
| `explanation-validator` | validator chip | `data-valid` = `true` / `false` |
| `reason-codes` | list, each `li` has `data-code` | |
| `prob-misuse`, `prob-foreign_senders`, `prob-payoff_max`, `prob-drop_consignee` | probability rows | `data-calibrated`, `data-raw` |
| `top-features` | feature vs baseline table | |
| `state-text-toggle` | expands "What the model read" | |
| `state-text` | monospace state text (only after expanding) | |
| `latency-breakdown` | latency bar and legend | `data-total` |
| `cost-table` | expected cost table | |
| `cost-row-<action>` | one row per action | `data-chosen`, `data-greedy` = `true` / `false` |
| `cost-note` | "Costs are assumptions" note | |
| `ask-panel` | ask panel | |
| `ask-input` | question textarea | |
| `ask-yes`, `ask-no` | meaning of yes and no | default to "yes"/"no" when empty |
| `ask-example` | fills the reshipping drop example | |
| `ask-submit` | POST /decisions/{id}/ask | |
| `ask-result` | latest answer | `data-probability`, `data-qid` |
| `analyst-panel` | analyst panel | |
| `analyst-existing` | existing label, if any | `data-label` |
| `analyst-note` | note textarea | |
| `analyst-confirm-fraud`, `analyst-mark-legit` | POST /decisions/{id}/analyst | |
| `analyst-result` | confirmation | `data-label` |
| `analyst-result-hash` | audit hash returned by the API | full hash text |
| `decision-audit-hash` | decision audit hash (shortened, full in `title`) | |

## Review queue (`/#/queue`)
| id | element | notes |
|---|---|---|
| `queue-filter` | action filter group | |
| `queue-filter-all`, `queue-filter-<action>` | filter buttons | `aria-pressed` |
| `queue-sort` | select: `misuse` (default), `carrier_cost`, `newest` | |
| `queue-refresh` | reloads | |
| `queue-table` | table | |
| `queue-row` | one per decision, click opens detail | `data-decision-id`, `data-action`, `data-misuse`, `data-carrier-cost` |

## Dashboard (`/#/dashboard`)
| id | element |
|---|---|
| `dashboard-kpi-bookings` | bookings scored |
| `dashboard-kpi-held` | shipments held |
| `dashboard-kpi-net-prevented` | net loss prevented (BRL), with prevented and friction shown |
| `dashboard-kpi-fpr-legit` | FPR on legit shippers |
| `dashboard-kpi-fpr-hard-negative` | FPR on hard negatives |
| `dashboard-kpi-latency` | p50 with p99 |
| `trend-chart` | trend chart wrapper, `data-mode` = `typology` / `held` |
| `trend-mode-typology`, `trend-mode-held` | series toggle |
| `trend-table-toggle`, `trend-table` | table view of the trend |
| `action-mix`, `action-mix-<action>` | action mix (`data-count`) |
| `dashboard-assumptions` | assumptions note |

## Learning (`/#/learning`)
Mock mode: labels start at 0 (scoring then confirming in the decision view adds analyst labels). Retrain takes about 2.5 s and returns 409 below 20 new labels. Retrain runs alternate: the 1st, 3rd, ... pass the gate and deploy; the 2nd, 4th, ... are rejected. So to show a rejected run: simulate, retrain (deployed), simulate again, retrain (rejected).

| id | element | notes |
|---|---|---|
| `learning-active-version` | active model version | text is the version |
| `learning-label-count` | labels since last retrain | `data-count` |
| `learning-source-<source>` | per-source count, e.g. `learning-source-analyst`, `learning-source-simulated_analyst` (badged Simulated) | `data-count` |
| `learning-simulate-n` | number input for simulated labels | |
| `learning-simulate-button` | POST /learning/simulate_feedback | |
| `learning-simulate-result` | confirmation after simulating | |
| `learning-retrain-button` | POST /learning/retrain (`min_new_labels` 20) | |
| `learning-retrain-progress` | shown while retraining, elapsed seconds | |
| `learning-result` | retrain result wrapper | `data-run-id` |
| `learning-result-table` | current vs candidate table; rows have `data-metric`; change cell child has `data-direction` = `better` / `worse` / `same` | |
| `learning-gate-check` | one per gate check | `data-name`, `data-passed` |
| `learning-deployed-status` | deployed or rejected banner | `data-deployed` = `true` / `false` |
| `learning-version-table` | version history | |
| `learning-version-row` | one per version, newest first | `data-version`, `data-active` |
| `learning-rollback-button` | non-active rows that once passed the gate, POST /learning/rollback | `data-version` |
| `learning-version-rejected` | non-active rows the gate rejected (no rollback offered) | |
| `learning-laya-note` | Laya offline retraining note with export path | |
| `learning-error` | error box (for example the 409 when too few labels) | |

## Audit (`/#/audit`)
| id | element | notes |
|---|---|---|
| `audit-verify-button` | re-runs GET /audit/verify | |
| `audit-status` | result | `data-state` = `checking` / `ok` / `broken` / `error`; `data-ok` |
| `audit-records` | record count | |
| `audit-head-hash` | head hash | |
| `audit-first-bad` | first bad index, only when broken | |

## Shared
| id | element |
|---|---|
| `error-box` | generic error box (default id) |
