---
id: cpu-saturation
title: CPU saturation (CPU hog, CPU stress)
applies_to: cpu
keywords: cpu, hog, stress, saturation, saturated, busy, throttling, throttled, contention, queueing, queue, slow, latency, runaway, loop
summary: One service's CPU jumps to a plateau, its own operations slow down, and its callers slow down with it because requests wait for CPU.
---

## Symptoms

- **Metrics.** The faulty service's `cpu` rises sharply at the onset, often to a flat plateau
  (the CPU limit, or what the stressor takes). Its `latency-50` and `latency-90` rise at the
  same time. Callers' latencies rise a little later or at the same time. Other services' CPU
  stays flat or drops slightly, since less traffic gets through.
- **Traces.** The faulty service's own spans get longer (callee side grows). Inside them the
  gap before its first child span (a database or downstream call) grows: the request waits for
  CPU before any work happens. Callers' client spans grow by about the same amount.
- **Logs.** Usually unchanged. Callers may log timeouts if the latency exceeds their deadlines.

## Confirm

1. `detect_anomalies`: a `cpu` anomaly going up with a high score (>= 10) and an early onset.
2. `query_metrics(service=<suspect>)`: CPU plateau from the onset; latency rising with it.
3. `query_traces(service=<suspect>)`: the suspect's own operations have a high p90 ratio. Open
   an `example_trace` and look for a long gap between the suspect's span start and its first
   child.
4. `get_service_topology`: the other slow services should be callers of the suspect.

## Distinguish from

- **A busy caller.** A service that retries or waits on a slow callee can use more CPU too, but
  its rise is smaller, and its callee shows the stronger change and the earlier onset.
- **More load.** If `workload` rises at the entry point and CPU rises across many services, the
  traffic grew; that is not a fault in one service.
- **Memory pressure.** Garbage-collection thrashing raises CPU as well; check whether `mem` rose
  first (see memory-pressure).
- **Network delay.** Latency rises without any CPU increase (see network-delay).

## Typical causes

A runaway loop or background job, a stress process in the container, CPU limits (throttling)
lowered by a configuration change, a noisy neighbour on the same node.
