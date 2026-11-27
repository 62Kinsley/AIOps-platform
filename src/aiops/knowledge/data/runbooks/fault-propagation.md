---
id: fault-propagation
title: How faults propagate between microservices
applies_to: any incident
keywords: propagation, cascade, upstream, downstream, caller, callee, victim, symptom, blast radius, dependency, onset, order, call graph
summary: Faults spread from the faulty service to the services that call it. Use the call graph and onset times to tell the cause from its victims.
---

## The basic rule

A request waits for every call it makes. When a service becomes slow or fails, every service
that calls it, directly or through others, becomes slow or fails too. So latency and errors
spread **towards the entry point** (to callers), while resource saturation (CPU, memory, disk,
sockets) stays **local** to the faulty service.

## Telling the cause from victims

- **Own resource anomaly.** The faulty service usually shows a resource anomaly of its own
  (CPU, memory, disk I/O, sockets), or new errors in its own logs. Victims show latency or error
  anomalies without one.
- **Position.** Among the services with latency anomalies, the cause is the one whose
  dependencies are healthy. If A calls B and both are slow, but B's own dependencies are fine,
  B is the better candidate.
- **Onset.** The cause's anomalies start first or at the same second; victims follow within
  seconds. Metrics are sampled about once a second, so near-equal onsets are not decisive.
- **Traces.** In a call A -> B, compare A's span (caller side) with B's span (callee side). If
  both grow, the time is spent inside B. If only A's grows, it is lost between them: network
  delay, or requests queued before B starts working (B saturated).

## Victim patterns that mislead

- **The entry point** (frontend, front-end, ts-ui-dashboard) shows every symptom of every
  fault.
- **Traffic drops.** When requests fail or time out early, the services after the failure
  point receive less traffic: their CPU, workload and log volume drop. Look for the boundary
  where traffic stops.
- **Retries and timeouts.** Callers that retry can show more CPU, sockets and error logs than
  the faulty service. Their errors usually name the callee ("upstream connect error", "failed
  to call ...", "connection refused").
- **Shared dependencies.** If two unrelated services slow down together, look for a dependency
  they share, such as a database or a commonly called service.

## Checklist

1. List anomalous services from `detect_anomalies`.
2. Draw their call relations from `get_service_topology`.
3. Find the anomalous service with no anomalous dependency; check its own resources and logs.
4. Confirm that every other anomalous service calls it, directly or indirectly.
