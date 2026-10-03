# Phase 3: Evaluation and demo design (FraudShield, Top-1 "Who is this account shipping FOR?")
Date: 2026-10-02. Legend: [V] verified this session (computed from the raw Olist CSVs or fetched source); [P1] carried from Phase 1 notes (sourced there); [A] our assumption / design choice; [U] unverified.

## 0. Olist facts verified this session (computed from raw files)
Source of files: GitHub mirror https://github.com/ckoliveiraa/pipeline-olist (raw/*.csv.gz), identical row counts to the Kaggle release as widely reported (99,441 orders, 112,650 items, 3,095 sellers). Kaggle page https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce did not render via fetch; license CC BY-NC-SA 4.0 is from Phase 1 search [P1]. Analysis scripts: scratchpad/olist/a.py, b.py, c.py.

Tables and exact columns [V]:
- orders (99,441): order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
- order_items (112,650): order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value
- products (32,951): product_id, product_category_name, product_name_lenght, product_description_lenght (sic), product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm
- sellers (3,095): seller_id, seller_zip_code_prefix, seller_city, seller_state
- customers (99,441): customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state
- order_payments (103,886): order_id, payment_sequential, payment_type, payment_installments, payment_value
- geolocation (1,000,163): geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state
- order_reviews (99,224): review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp
- plus product_category_name_translation

Date range [V]: purchase 2016-09-04 to 2018-10-17, but volume is effectively 2017-01 to 2018-08 (2016: 316 shipments; Sept 2018: 1; Oct 2018: stragglers, mostly canceled). Approved_at 2016-09-15 to 2018-09-03. 96,478 delivered.
Shipment unit: one (order_id, seller_id) pair = one booking. 100,010 bookings; 1,278 orders have >1 seller [V].

Seller order-count distribution (bookings per seller, whole period) [V]:
| threshold | sellers | share of all bookings |
|---|---|---|
| >= 1 | 3,095 | 100% |
| >= 5 | 1,794 | 97.4% |
| >= 10 | 1,271 | 93.9% |
| >= 20 | 818 | 87.7% |
| >= 30 | 634 | 83.3% |
| >= 50 | 429 | 75.5% |
| >= 100 | 210 | 60.0% |
| >= 200 | 91 | 43.5% |
| >= 500 | 26 | 24.5% |
| >= 1000 | 10 | 13.9% |
Median 6, p75 21.5, p90 69.6, max 1,854. Before 2018-03-01: 549 sellers with >= 20 bookings, 267 with >= 50. Sellers active in test window (Jun-Aug 2018) with >= 20 prior bookings: 562, carrying 12,833 of 19,263 test bookings.
Verdict: per-account histories are rich enough for ~550 accounts (the victim pool for takeover typologies), thin for the long tail. The long tail is realistic (most carrier accounts are small) and is where the "thin history" path of the model must be tested.

Other distributions used to anchor injection [V]:
- freight_value per booking (BRL): median 17.06, p90 38.89, p99 103.1, mean 22.5; total R$2.25M.
- Shipment weight: median 750 g, p90 6.55 kg, p99 22.35 kg. Product dims median 25x13x20 cm.
- Seller-to-customer distance (zip-prefix centroids): median 434 km, p90 1,458 km, p99 2,483 km.
- Linear freight model: freight ~= 8.78 + 2.89 * kg + 11.52 * (km/1000) BRL; corr(freight, weight) 0.64, corr(freight, km) 0.32. Used to price injected bookings, plus an empirical residual sampled from the same weight/distance bin.
- payment_type (customer side): credit_card 76,795, boleto 19,784, voucher 5,775, debit_card 1,529.
- Sellers in 23 states (SP 1,849), customers in 27; 2,246 distinct seller zip prefixes; 14,976 destination zip prefixes.
- 35.9% of bookings are same-state.
- Lead times: approved -> carrier handoff median 43 h; handoff -> delivered median 7.1 days; estimated delivery window median 23 days.
- Daily bookings: ~234/day Jan-May 2018, ~214/day Jun-Aug 2018; Black Friday 2017-11-24 = 1,175 (peak day of the dataset).
- Black Friday surge per seller (24-26 Nov 2017 daily rate vs Oct 2017): median 5.2x, p90 15.5x (396 sellers).
- Max/median weekly volume for sellers with >= 50 bookings: median 4.0x, p90 7.0x. Volume spikes are normal.
- KEY FINDING: established sellers (>= 20 prior bookings) send 94.9% of bookings to a destination zip prefix they never used before; median seller reaches 12 destination states over its life. Consignee novelty is NOT a fraud signal in a marketplace. The "customer-base shift" signal must be built on the payer -> sender/origin side and the package profile (category, weight, value-per-label), plus consignee IN-DEGREE across accounts, not on consignee novelty per account. This changes Phase 2 wording "consignees it never served": drop it for this data.
- Repeat consignees: 3,262 customer_unique_ids received from >= 2 distinct sellers, 280 from >= 3, 16 from >= 5, max 10. These are the real hard negatives for the mule-drop typology.
- High-value categories available for realistic injected contents: telefonia 1,134 products, informatica_acessorios 1,639, relogios_presentes 1,329, eletronicos 517, consoles_games 317, pcs 30.
- Data quirks: 14 bookings missing approved_at, 1,010 missing carrier handoff; 6 products with weight 0/null; shipping_limit_date has 4 values after 2018-12 (max 2020-04-09, bad data).

## 1. Data plan: Olist as a carrier booking stream
### 1.1 Field mapping (REAL fields)
| Carrier concept | Olist source | Notes |
|---|---|---|
| booking_id | order_id + seller_id | 100,010 bookings |
| shipper account (account holder) | seller_id | real |
| registered origin | seller_zip_code_prefix -> geolocation centroid | real, one origin per seller |
| consignee | customer_unique_id; dest = customer_zip_code_prefix | real |
| booking time | order_approved_at (fallback purchase ts) | [A] label is created once the seller is obliged to ship; shipping_limit_date is the ship-by deadline, kept as "promised induction deadline" |
| first scan / induction | order_delivered_carrier_date | real; drives the scan-gated action and label maturity |
| delivery | order_delivered_customer_date | real; drives the post-delivery baseline |
| package weight/dims | sum of product_weight_g; product dims (volume summed; max single dim kept) | real catalog values, not measured |
| contents category | product_category_name | real |
| billed shipping cost | freight_value summed per booking | real; the customer pays it to the marketplace, we use it as the carrier's billed amount [A] |
| declared value | price summed per booking | real |
| service level proxy | (estimated_delivery - purchase) days vs lane median: "express" if < lane p25 | derived [A] |
| distance | haversine between zip-prefix centroids | derived |

payment_type is the BUYER's instrument (one payer per order, the customer). It is not the carrier payer. Use it only as a booking context attribute (e.g. boleto vs card) and do not present it as payer identity.

### 1.2 Synthetic but documented billing layer [A, SYNTHETIC]
Real-world assumption: on carrier B2B accounts the shipper account holder is billed. Olist's sellers ship via Olist's logistics partners [V via search summary of dataset description], so the real default is payer = shipper. We add a generated layer for every account, legit and fraud alike, with one fixed seed, so it carries no label information:
- billing_account_id (= seller_id for legit), account_number (random 9 chars), account_created_at = first booking minus Uniform(30, 365) days for sellers present before 2017-03, else first booking minus Uniform(1, 30) days.
- payer_instrument_id: 1 stable instrument per account (type drawn ACH 60% / card 40% [A]); 5% of legit accounts get a second instrument at a random date (legit change, hard negative).
- login_ids: 1-3 per account (Poisson(1)+1), each with device_id and channel (API 50% / web 50% for accounts >= 50 bookings, web 90% otherwise) [A]. 3% of legit accounts link a new login at a random date (legit new employee, hard negative).
- sender_origin: equals registered origin for all legit bookings. Third-party billing (payer account != sender origin) does not occur in real rows; T1 creates it. To avoid "any third-party billing = fraud", inject legit third-party billing too: 2% of legit established accounts get a second legit origin (a real zip in the same state, "new warehouse") from a random date onward [A, hard negative].
- billing_correction history: per account synthetic rate ~ Beta(1, 30) (mean ~3%) [A; the "3-8%" vendor figure is unverified]; T6 raises it.
Every synthetic column is listed in the data card with generator, distribution and seed.

## 2. Injection protocol
General rules:
- Rule-based, never GAN/CTGAN (Sajja 2026, arXiv 2604.13125: synthetic generators lose burst/multi-account motifs [P1]).
- Injected rows reuse REAL values wherever possible: real product rows (weight, dims, category, price), real zip prefixes present in geolocation, real customer_unique_ids or new ids drawn with the same format, timestamps drawn from the real hour-of-day and day-of-week distribution of the account (or the global one), handoff and delivery lags sampled from the empirical lag distribution of the same lane (state pair).
- freight_value for injected rows = linear model above + empirical residual from the same (weight bin, distance bin).
- Every injected row has: is_injected=1, scenario_id (T1..T6), campaign_id, camouflage_level (0/1/2), params_json, seed. Real rows untouched (is_injected=0). Label = is_injected and fraud-typology; real rows are assumed legit [A, caveat: Olist could contain undetected fraud; impact is small label noise].
- Campaigns may not cross split boundaries (onset and end must fall in the same split with a 14-day embargo at each boundary).
- Victim accounts are disjoint across train/val/test; seller_id is never a model feature.
- Master seed 20261002; campaign seed = master + 1000*typology + 100*split + replicate. 5 injection replicates (r=0..4); all headline numbers = mean and sd across replicates, CIs by campaign-level cluster bootstrap (1,000 resamples), because campaigns, not rows, are the independent units.

### Typologies (parameters, anchors)
T1 label-resale takeover (I2, I5; Harrod DOJ D.D.C., Hao et al. "single account pays for multiple different labels")
- Victim: real seller with >= 20 prior bookings.
- New login linked to victim at onset (synthetic login layer), probability 0.8; camouflage 2 sets 0 (uses existing login, e.g. stolen password).
- Phase A own-use: 3-14 days, 1-3 bookings/day, single new origin (a real zip prefix, other state with p=0.7), packages from one category (calcados/shoes analogue: fashion_calcados or esporte_lazer real products) [A, mirrors Harrod's shoe resale].
- Phase B resale: daily volume ramps linearly from 2 to target V ~ Uniform(1.5, 6) x victim's own p90 daily volume (cap 40/day), duration 7-45 days; each booking's origin from a pool of K ~ Uniform(5, 40) distinct real zip prefixes (third-party senders), consignees random real customer zips (population-weighted), contents sampled from the global catalog (not victim's categories).
- Cutoff and switch: with p=0.5 the same fraud ring moves to a second victim account, reusing >= 50% of the sender origin pool and device ids (graph link) [Harrod: "obtained a second company's account"].
- Victim's genuine bookings continue interleaved (real rows).
- Daily counts are our parameterization; DOJ gives only ">$900K" and "thousands of labels" [P1].

T2 bust-out new account (I3; Butt DOJ S.D. Tex., Grizzle DOJ W.D. Tenn.)
- New synthetic account, account_created_at 0-3 days before first booking, registered origin a real zip; name impersonation not modeled (no names in data).
- 10-60 bookings over 7-30 days, high-value categories (telefonia, informatica_acessorios, eletronicos, consoles_games, relogios_presentes), declared value from real prices of those products, long distance (>= p75 km) with p=0.7.
- Ends when "billing fails": stop at day D ~ Uniform(7, 30).
- Link to previously defaulted T2 accounts via shared payer_instrument_id or device_id: p=0.6 at camouflage 0, 0.2 at level 1, 0 at level 2.
- International destination (Dubai in Butt) cannot be represented in Olist (domestic only) [limitation].
- Hard negatives: the 537 real sellers whose first booking falls in the test window (2,487 bookings).

T3 mule-drop consignee cluster (I4; Hao et al. CCS 2015, Krebs 2023)
- Drop = new consignee id at a real residential zip prefix; first package 2-7 days after "recruitment"; 5-15 packages per drop (outliers to 50 with p=0.05); active ~30 days (Uniform(21, 35)).
- Packages to drops come from M distinct sender accounts (taken-over legit accounts or T2 accounts), M ~ Uniform(3, 10); contents: Apple-like phones and electronics dominate in Hao (Apple 57%, cameras ~20%): sample from telefonia 55%, eletronicos/informatica 25%, other high-value 20% [anchored proportions].
- Destination concentration: >= 85% of a ring's drops in one metro area (Hao: >= 85% to Moscow area); we use one Brazilian metro picked at random per ring (e.g. a single 2-digit zip region) [A, domestic adaptation].
- Rings of 5-20 drops.
- Hard negatives: 280 real consignees receiving from >= 3 distinct sellers (16 from >= 5).
- Held out from training (Section 3).

T4 payoff-maximizing use of a taken-over account (I7; LabelsBank $2 flat price, NullShip $3-9)
- Victim: established real seller. Bookings at roughly the victim's normal rate (Uniform(0.8, 1.5) x median daily), so no volume spike.
- Profile shift: weight drawn from the real catalog above the victim's own p90 weight, distance above victim's p90 km, express service proxy with p=0.7. Freight per label therefore lands at > p95 of the victim's history.
- 10-40 bookings over 5-21 days.
- Tests the "value extracted per label" feature directly; the fraudster's payoff = billed freight minus flat resale price (set R$10 [A]).

T5 test-then-burst (HYPOTHESIS, unsourced for carriers; labeled as such everywhere)
- Victim: established seller. 1-3 test bookings that look normal for the victim (sampled from the victim's own history: category, weight, lane) at off-hours, then gap 1-7 days, then burst of 10-30 bookings in 1-3 days with T1-style counterparties.
- Used only to test the sequence part of the state; no claims about prevalence.

T6 declared weight/dims manipulation (I8; Kavanaugh case for value inflation; UPS billing-correction audit fee)
- Victim or new account; after onset, 30-70% of bookings declare weight = true x Uniform(0.3, 0.7) and dims shrunk similarly; true values revealed at first scan (handoff timestamp).
- Account billing_correction rate history rises accordingly.
- Expected behavior: weak at booking, caught at the scan gate. This typology evaluates the "scan-gated allow" action, not booking-time detection alone.

### Injection rates
- Target prevalence 1.5% of bookings per split (test: ~290 fraud bookings vs 19,263 real), and a low-prevalence sensitivity setting 0.3%.
- Mix by bookings: T1 35%, T4 15%, T2 15%, T3 15%, T5 10%, T6 10%. In train/val, T3 is replaced by more T1/T4 (held out).
- Campaign counts in test (approx.): T1 8 campaigns, T4 5, T2 6, T3 2 rings (~20 drops), T5 4, T6 4. Accounts touched: ~25 victims of 562 established test accounts (4.4%).
- Camouflage levels in test: 50% level 0, 30% level 1, 20% level 2; results are reported per level.

### Camouflage levels (apply to T1, T2, T4, T5)
- Level 0: as specified.
- Level 1: origins/consignees restricted to the victim's own states; hours matched to the victim's hour distribution; no new login.
- Level 2: Level 1 + contents sampled from the victim's own category mix + volume ramp capped at the victim's p90 weekly volume + no graph links to other campaigns. Only the counterparty set (sender origins) and payoff shift remain.

### Data card (one file, data_card.md, generated by the injector)
Source and license (Olist CC BY-NC-SA 4.0, non-commercial); row counts real vs injected per split and typology; every synthetic column with generator and seed; parameter table above with source per parameter; what is real (all shipments, products, geography, timing), what is synthetic (billing layer, logins, devices, all fraud rows), known gaps (domestic only, no device/IP, no measured weights, service tier is a proxy, one origin per legit seller), intended use and misuse warnings.

## 3. Time-based split (booking_ts = approved_at)
- Warm-up history only (no labels used): 2016-09-04 to 2016-12-31 (316 bookings).
- Train: 2017-01-01 to 2018-02-28 (~59,300 bookings real). Includes Black Friday 2017.
- Validation: 2018-03-01 to 2018-05-31 (21,429). Used for early stopping, per-question temperature fit (<= 400 held-out per Laya notebook), threshold and conformal calibration, cost-matrix tuning.
- Test: 2018-06-01 to 2018-08-31 (19,263). Touched once per model.
- Excluded: Sept-Oct 2018 stragglers.
- 14-day embargo: no injected campaign starts within 14 days before a split boundary.
- Features for any booking use only data with timestamps before that booking's booking_ts, plus label maturity: a booking's fraud label becomes visible to the online learner at handoff + 7 days (median transit) [A, mirrors post-delivery discovery].
- Second fold for seasonality (hard negatives): train 2017-01-01 to 2017-10-15, test 2017-11-01 to 2017-12-31 (Black Friday). Report FPR on legit bookings on 2017-11-24..27 specifically.

Held-out typology: T3 (mule drop).
Why: (a) its signal lives on a different side of the graph (consignee in-degree across accounts) than T1/T4 (payer -> origin diversity, payoff), so success means the representation generalizes, not that the model memorized a near-duplicate; T4 is close to T1 (taken-over account, profile shift) and would be a weak hold-out; (b) its parameters are the best-sourced (Hao et al.), so the test is meaningful; T5 is unsourced, holding it out would prove little; (c) real hard negatives exist (280 multi-seller consignees). Secondary: leave-one-typology-out sweep with LightGBM (cheap, 6 runs) and with Laya for T3 and T4 only if GPU time allows.

## 4. Metrics (all on test, per injection replicate, then mean +- sd; CIs by campaign cluster bootstrap)
1. PR-AUC (average precision, sklearn average_precision_score) on booking-level labels; also per typology (one-vs-legit) and per camouflage level. Report prevalence next to it.
2. Precision at fixed review capacity. Test volume is ~214 bookings/day, so 0.5% = ~1 booking/day, too small to be stable. Use: capacity = top 1% of bookings per day (about 2 reviews/day at Olist scale; scale up for a real carrier) and also top 0.5% and 2% over the whole test window. Compute by ranking within each day by score (or expected loss), taking top-k per day, precision = fraud among selected / k. Also recall at the same capacity.
3. FPR on legitimate shippers: FP / legit bookings at the operating threshold of each action (hold or block; review counted separately). Report also account-level FPR = share of legit accounts with >= 1 hold in the test window. Specifically on hard-negative slices: (a) 624 real bookings where an established seller ships to a new state for the first time (323 sellers); (b) 2,487 bookings of the 537 real new sellers; (c) Black Friday 2017 fold, 2017-11-24..27; (d) synthetic legit changes (new warehouse origin, new login, new instrument); (e) the 280 multi-seller consignees.
4. Revenue loss prevented (BRL): Prevented = L_base - L_ours - F.
   - L_base = sum freight_value of fraud bookings booked before the baseline detection time of their campaign (all of them are lost under the baseline).
   - L_ours = sum freight of fraud bookings our policy lets into the network: action allow, or scan-gated allow where the scan check does not fire (scan fires for T6 weight mismatch > 20%, and for any booking whose induction location differs from booked origin; other typologies pass the scan) [A]; owner-confirm counts as stopped with p=0.9 (owner denies) [A].
   - F = friction: c_review x reviews + c_hold_FP x legit holds + c_confirm x legit owner-confirms + c_block_FP x legit blocks. Defaults [A]: c_review R$5, c_confirm R$1, c_hold_FP = R$5 + 1 day delay penalty (10% of freight), c_block_FP = freight + R$20 churn proxy. Sensitivity table at 0.5x, 1x, 2x. Do not convert BRL to USD in claims.
5. Detection lead time. Baseline "post-delivery pattern analysis" (B-post): a weekly batch every Monday 00:00 over bookings DELIVERED by then (delivered_ts from real lane lags), applying account-level rules: account flagged if in the last 28 days of delivered bookings it has >= 3 new sender origins, or freight-per-label > account p99 on >= 3 bookings, or a consignee received from >= 3 accounts, or (new account < 30 days and >= 5 high-value bookings). The same rules applied at booking time form the "rules only" baseline. Lead time per campaign = (# fraud bookings before B-post flags the account) - (# fraud bookings before our system first holds/blocks/confirms one), and the same in days. Report median and IQR across campaigns, plus the share of campaigns our system catches before their k-th booking for k = 1, 3, 5, 10. Secondary baseline: the first booking where LightGBM-booking crosses its threshold.
6. Calibration: ECE with 15 equal-width bins (Guo et al. 2017 convention), per Laya question and for P(fraud); plus Brier score and adaptive (equal-mass) 15-bin ECE because positives are rare and equal-width bins put almost all mass in bin 1. Reliability diagram with bin counts. Report before and after temperature refit.
7. Latency: per-booking end-to-end (feature fetch + state serialization + model + decision + audit write), p50/p95/p99 over 10,000 replayed test bookings at 1, 10, 50 req/s, hardware stated (T4 or the demo laptop). Laya-only latency reported separately (vendor reports ~39.5 ms per question on T4 [P1]; we measure our own). Explanation generation is async and excluded from the decision path; report its latency separately.
8. Explanation quality.
   - Faithfulness (automatic): the explainer emits JSON claims, each with a field reference (e.g. features.novel_origins_7d = 14, laya.q_third_party_billing.p = 0.91). Checker: every claim references an existing field and the stated value matches (exact for categorical, +-1% numeric). Faithfulness = share of explanations where ALL claims pass. Plus a deletion test: replace the top cited field with the account's baseline value, rescore; share of explanations where P(fraud) drops by >= 0.1.
   - Analyst rubric: 3 team members, blind to model, score 50 held explanations (mix of TP/FP) 1-5 on correctness, usefulness for the decision, plain language; report mean and Krippendorff's alpha. Small n, report as indicative.

Baselines and systems (same features where applicable, same splits):
| ID | System |
|---|---|
| B0 | Rules only (the B-post rules applied at booking time) |
| B1 | LightGBM on per-booking features (weight, dims, km, freight, category, hour, service proxy, account age) |
| B2 | LightGBM + account-change and billing-graph features (counterparty diversity, novel origins, payoff z, consignee in-degree, login/instrument change) |
| B3 | Stock Laya zero-shot on the serialized state (expect weak; vendor zero-shot below majority [P1]) |
| B4 | Jev zero-shot via API (only if access granted; else "n/a, waitlist") |
| M1 | Fine-tuned Laya on serialized state |
| M2 | Fine-tuned Laya + B2 signals in the state |
| M2+P | M2 + cost-aware action policy (for loss/friction metrics) |

Results table template (fill per replicate mean +- sd):
| System | PR-AUC all | PR-AUC T3 (held-out) | PR-AUC cam-2 | P@1%/day | R@1%/day | FPR legit (hold) | FPR hard-neg new-state | FPR new sellers | FPR BF 2017 | ECE-15 | Brier | Lead time (bookings, median) | Loss prevented R$ (net) | p50 / p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 Rules | | | | | | | | | | n/a | n/a | | | |
| B1 LGBM booking | | | | | | | | | | | | | | |
| B2 LGBM + signals | | | | | | | | | | | | | | |
| B3 Laya zero-shot | | | | | | | | | | | | | | |
| B4 Jev zero-shot | | | | | | | | | | | | | | |
| M1 Laya FT | | | | | | | | | | | | | | |
| M2 Laya FT + signals | | | | | | | | | | | | | | |
Honesty rule: if B2 >= M2 on PR-AUC and ECE, say so and argue Laya only on what it actually wins (e.g. held-out typology, calibration per question, multi-question outputs for the action layer); do not hide it.

## 5. "Is injected fraud too easy?" checks
1. Rules ceiling: if B0 reaches PR-AUC >= 0.95 or recall >= 0.95 at 1%/day on level-0 test, the injection is too easy; raise camouflage share and re-run. Report B0 on all camouflage levels.
2. Artifact / adversarial validation: train LightGBM to separate injected vs real rows using ONLY non-semantic fields (timestamp seconds, id formats, missingness pattern, zip validity, price decimals, residual of freight model). Target AUC <= 0.55; anything higher means a leak, fix the generator.
3. Camouflage levels 1 and 2 reported separately; the claim rests on level 2.
4. Held-out typology T3 and the LOTO sweep.
5. Feature ablation on B2 and M2: remove (a) counterparty-shift group, (b) payoff group, (c) graph group, (d) login/instrument (synthetic) group. If removing only (d) collapses performance, the result depends on synthetic fields and must be reported as such.
6. Hard-negative FPR (Section 4.3) next to every headline number.
7. Real-label sanity for the graph component: run the same graph feature pipeline / GNN on GADBench Amazon (11,944 nodes, 9.5% fraud) and YelpChi (45,954, 14.5%) [P1] and compare AUC/AP to GADBench published numbers; and on an IEEE-CIS entity graph (card1/addr1/P_emaildomain/DeviceInfo shared-entity edges, 590,540 tx, 3.5% fraud) with a time split on TransactionDT. Purpose: show the graph code is not tuned to our own injection.
8. Sensitivity to injection parameters: re-run test with T1 K (origin pool) halved and volume ramp halved; plot PR-AUC vs parameter.
9. Adaptive fraudster stress test (RL component): attacker policy chooses origin pool size, ramp and contents under a payoff constraint against frozen M2+P; report payoff achievable vs level-2 handcrafted. Optional if time.

## 6. Five-minute demo storyline
Prep: replay server on test window (2018-06..08) with injected campaigns, models warm (LAYA_PRELOAD=1), explanation cache warmed for scripted bookings, dashboard pre-aggregated.
- 0:00-0:35 Hook. "In 2020 two men got a company's UPS login. First they shipped their own shoes. Then they sold labels to strangers. Over $900,000 before anyone noticed (DOJ, US v. Harrod). USPS OIG this June: about $3.1B lost to counterfeit labels in two years, and 97% of those parcels were still delivered. Detection after delivery is too late. We score at booking." One sentence on data: real Olist shipments, documented injected fraud.
- 0:35-1:05 Legit booking: established seller, usual lane. Shows ALLOW, score, latency in ms (live measured number), "why not suspicious" one line.
- 1:05-1:45 Hard negative: same kind of seller, first ever shipment to a new state (real row). Decision: SCAN-GATED ALLOW, not block. Show that 95% of such sellers' bookings go to new zips, so novelty alone means nothing.
- 1:45-2:45 T1 takeover in phase B: payer account suddenly billing 14 new sender origins in 7 days, heavy long express parcels. HOLD. Show per-question Laya probabilities (e.g. "consistent with account history?" no 0.9x; "third-party billing pattern?" yes 0.8x; "payoff above account norm?" yes), and the plain-language explanation with every claim linked to a field. Show "first held at booking #4 of the campaign; weekly post-delivery batch would flag it at booking #N, Monday X".
- 2:45-3:25 Analyst confirms fraud in the queue. Feedback written to label store; show OPE panel: SNIPS estimate of the proposed new threshold policy from logged exploration (with CI), explicitly "evaluation, not online learning on the fly".
- 3:25-4:05 Dashboard: daily held shipments, trend by typology, revenue loss prevented (net of friction) with the cost assumptions visible, FPR on hard negatives.
- 4:05-4:25 Audit log: open the held booking's record (model version, input state hash, probabilities, action, explanation), run verify on the hash chain, then tamper one byte in a copy and show verify failing.
- 4:25-5:00 Results table (B0..M2), held-out T3 row, camouflage-2 row, and honest limitations: injected fraud, domestic only, synthetic billing/login layer, BRL freight from a marketplace not a carrier, small analyst rubric, Jev status.

Failure plan:
- Pre-recorded full 5-minute screen capture (1080p, local file, not streamed) on the demo laptop and a USB stick; switch within 10 s if anything stalls > 5 s.
- Replay mode: the scripted bookings have cached responses (scores, probabilities, explanation JSON) keyed by booking_id; UI has a "cached" badge so it is never presented as live when it is not.
- Laya server down: fall back to CPU model (state latency honestly), or cached.
- Explanation LLM down or slow: template explanation from the same JSON claims (it is grounded either way).
- No network: everything runs locally; Jev comparison shown only as precomputed numbers.
- Results table and data card as static slides.
- Rehearse 3 times with a timer; one person drives, one narrates.

## 7. Open risks / unverified
- [U] Olist license string not re-read on Kaggle this session (page not fetchable); Phase 1 search says CC BY-NC-SA 4.0.
- [A] freight_value is paid by the buyer to the marketplace, used here as the carrier's billed amount.
- [A] all cost-matrix and scan-gate catch rates; shown with sensitivity.
- [U] Brazil truckers' strike (late May 2018) may distort lags at the end of validation; check lag distribution in May 2018 before fitting lag samplers.
- With ~290 fraud bookings and ~30 campaigns in test, CIs will be wide; campaign-level bootstrap is required, and claims of small differences between M2 and B2 should not be made.
