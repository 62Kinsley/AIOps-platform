---
id: network-delay
title: Network delay (added latency on a service's network)
applies_to: delay
keywords: network, delay, latency, slow, lag, rtt, round trip, tc, netem, jitter, slow network, response time, caller side
summary: Calls to and from one service take longer by a roughly constant amount while its CPU and memory stay normal; traces show the time lost between caller and callee.
---

## Symptoms

- **Metrics.** The faulty service's latency rises, and so does the latency of every service
  that calls it, by a similar, roughly constant amount. Its CPU does not rise; it often drops a
  little because fewer requests complete per second (`workload` may drop too). No other
  resource metric changes.
- **Traces.** This is the clearest signal. For calls **into** the faulty service, the caller's
  span grows while the callee's own span grows much less: the time is lost on the network. For
  calls **from** the faulty service to its dependencies, its client spans grow while the
  dependencies' own spans stay flat. The faulty service is the common end of all the edges
  whose caller-side latency grew.
- **Logs.** Usually quiet; callers with tight deadlines may log timeouts.

## Confirm

1. `detect_anomalies`: latency anomalies in the suspect and its callers, with no CPU, memory,
   disk or socket anomaly in the suspect.
2. `query_traces(service=<suspect>)`: in `calls`, compare `caller_p90_ratio` with
   `callee_p90_ratio` on the edges that involve the suspect. Caller side up, callee side flat
   means delay on the path.
3. Check whether the suspect's own outgoing calls are slowed too: delay on a pod's network
   slows both directions.

## Distinguish from

- **CPU saturation.** The callee's own span grows and its CPU rises (see cpu-saturation).
  Queueing at a saturated callee can also make caller-side latency grow more than callee-side
  latency, so check CPU before blaming the network.
- **Packet loss.** Loss produces mostly tail latency (p90 and above) and errors, not a constant
  shift of the median (see packet-loss).
- **A slow dependency of the suspect.** If the suspect's callee has its own resource anomaly,
  the cause is further down.

## Typical causes

Traffic shaping or a misconfigured network policy, a congested node or link, cross-zone
routing, a sidecar proxy in trouble, an injected delay.
