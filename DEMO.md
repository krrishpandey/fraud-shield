# Tracd Verify: demo video (3 to 5 minutes)

Record the screen with **Win + Alt + R** (Xbox Game Bar; turn the microphone on in Win + G so the phone call is heard),
or with OBS. Put your phone on **speaker** next to the laptop so the call is in the recording. Every call in the app also
has a **Play recording** button (Bolna's own recording), which you can play during the video instead.

## Before you press record
1. Start the app: `cd F:\PROJECTS\fraud-shield\fraudshield` then
   `%USERPROFILE%\.venvs\fraudshield\Scripts\python.exe -m uvicorn fraudshield.api.app:create_app --factory --port 8080`
2. Open http://localhost:8080, sign in with your name. Wait 30 s (real bookings load in the background).
3. Check the top of **Owner calls** says **Real phone calls**.
4. Phone: on speaker, not on silent. Balance in Bolna: each call is about $0.06.

## Script

| Time | Show (exact clicks) | Say |
|---|---|---|
| 0:00-0:30 | Title slide: "Tracd Verify" | "In shipping, fraudsters book parcels on other businesses' accounts. The owner finds out weeks later, from the bill." |
| 0:30-1:00 | **Try a booking**: click the **Established seller** example card. The Decision page shows **Allow** and the time ("decided in ... ms"). | "Tracd scores every booking before the label prints, in milliseconds. This one is normal: the label prints." |
| 1:00-1:20 | **Live bookings**: the summary line (Allowed / Blocked / Held) and the table. | "Every booking is allowed, blocked, or held. Held ones get a phone call." |
| 1:20-2:20 | **Owner calls** → filter **Demo calls** → **Play the demo**. Phone rings: answer on speaker, say **"No, that was not me."** Click the new row: transcript, **Stopped**, account locked. | "A 2 am booking from a new device is held, and Tracd calls the real owner. The owner says no: the parcel never gets a label." |
| 2:20-2:50 | In the same panel: **Why it was held** (Laya's plain-English reasons) and the **Play recording** button. | "The analyst sees why it was held in plain English, the full transcript and the recording." |
| 2:50-3:40 | Second call rings (honest shipper). Say **"Yes"**, pickup **"Plot 14, SIPCOT, Hosur"**, change **"We opened a new warehouse in Hosur."** Row turns **Released**. | "An honest shipper with a new warehouse: released in about a minute instead of waiting hours for an analyst. The new warehouse stops triggering warnings for 30 days." |
| 3:40-4:10 | Third demo booking: **Not called** (number changed 6 days ago) → **Needs a person**. | "If the phone number changed recently, we don't call: a fraudster could have changed it." |
| 4:10-4:40 | **Audit log** → **Verify chain**: "Chain intact". | "Every decision and every call is a tamper-evident record." |
| 4:40-5:00 | Closing slide | "Tracd Verify: stop fraud, release honest shippers fast." |

## Say it honestly
- The bookings are real Olist data (Brazil); the company and city names on the three demo bookings are demo labels.
- LightGBM decides; Laya explains (it scored lower than LightGBM in testing: 0.356 vs 0.777 PR-AUC).
- The phone calls are real Bolna calls to our own verified numbers (Bolna trial accounts can only call verified numbers).
