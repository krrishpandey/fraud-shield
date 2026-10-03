# Live stream load test

Measured 2026-10-03 16:46 on Windows 11, AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, 16 CPU threads. In-process FastAPI app, Laya in cached mode
(decisions from the LightGBM backup score), concurrency 4, about 60.0 s per rate.
Bookings are the seed-0 test window replayed in booked_at order. Latency is the service's own
end-to-end scoring time per booking (latency_ms.total), before the label would be issued.

| Target rate (/s) | Bookings | Achieved (/s) | Errors | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| 1.0 | 60/60 | 1.0 | 0 | 19 | 33 | 37 |
| 5.0 | 300/300 | 4.9 | 0 | 23 | 36 | 40 |
| 20.0 | 1200/1200 | 19.7 | 0 | 20 | 27 | 34 |
| 50.0 | 3000/3000 | 37.1 | 0 | 24 | 41 | 59 |
| 100.0 | 6000/6000 | 37.7 | 0 | 23 | 39 | 85 |

The service saturates at about 37.7 bookings per second on this machine: from a target of 50.0/s, the achieved rate stays below the target.
