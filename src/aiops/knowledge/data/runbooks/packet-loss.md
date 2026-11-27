---
id: packet-loss
title: Network packet loss
applies_to: loss
keywords: packet loss, loss, dropped packets, retransmission, retransmit, timeout, reset, connection reset, unavailable, deadline exceeded, tail latency, p90, errors, flaky
summary: Requests to or from one service intermittently stall or fail; tail latency and error rates rise far more than median latency.
---

## Symptoms

- **Metrics.** The faulty service's `latency-90` rises much more than its `latency-50`: most
  requests are fine, the unlucky ones wait for TCP retransmissions. Error rates (`error`) of
  the service and its callers rise; throughput (`workload`) may drop. CPU and memory are
  unchanged or lower.
- **Traces.** Calls into and out of the service show a heavy tail (p90 grows far more than
  p50) and, where status codes exist, errors such as gRPC UNAVAILABLE (14) or DEADLINE_EXCEEDED
  (4). Some calls may disappear (`calls_stopped_after_alert`) when connections fail.
- **Logs.** Callers log timeouts, "connection reset", "upstream connect error",
  "Unavailable", or retries; the faulty service may log failed outgoing calls.

## Confirm

1. `detect_anomalies(metric="latency")`: compare the p50 and p90 scores of the suspect and its
   callers; a much stronger p90 anomaly suggests loss.
2. `query_traces(service=<suspect>)`: p90 ratio much larger than the p50 change, error rates up
   on the suspect's edges.
3. `search_logs(query="timeout|reset|unavailable|refused")`: which service the errors name.

## Distinguish from

- **Network delay** shifts the median as well as the tail and rarely causes errors.
- **Socket exhaustion** also causes connection errors, but the faulty service's `socket` metric
  rises (see socket-exhaustion).
- **A crashed service** stops responding entirely: its spans stop and its log volume drops to
  zero, rather than a fraction of requests failing.

## Typical causes

A faulty network interface or link, an overloaded node or proxy, a misconfigured MTU, an
injected loss.
