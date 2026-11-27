---
id: socket-exhaustion
title: Socket and connection exhaustion
applies_to: socket
keywords: socket, sockets, connection, connections, file descriptors, too many open files, ephemeral ports, connection pool, refused, exhaustion, leak, keepalive
summary: One service holds far more sockets than usual; new connections to or from it fail or wait, so it and its callers see errors and latency.
---

## Symptoms

- **Metrics.** The faulty service's `socket` count jumps at the onset (often several times its
  baseline) and stays high. Its latency rises and errors appear at it or at its callers. CPU
  may rise a little.
- **Traces.** Calls to the service slow down or fail (connection setup waits or is refused);
  if it cannot open outgoing connections, its own calls to dependencies fail.
- **Logs.** "Too many open files", "Cannot assign requested address", "connection refused",
  "connection pool exhausted", or timeouts, in the faulty service or in its callers.

## Confirm

1. `detect_anomalies(metric="socket")`: one service with a strong increase at the onset.
2. `query_metrics(service=<suspect>)`: socket count plateau, latency and errors from the same
   time.
3. `search_logs(query="open files|refused|pool|timeout")`: errors in the suspect or naming it.

## Distinguish from

- **Callers that retry.** A caller of a failing service can open more connections as it
  retries. Its socket rise follows the callee's failure; check onsets and which service the
  errors name.
- **More load.** Socket counts rise across many services together when traffic grows.
- **Packet loss.** Errors without a socket increase point to the network (see packet-loss).

## Typical causes

A connection leak (connections opened and never closed), a missing connection pool, keepalive
misconfiguration, a file-descriptor limit set too low, an injected socket stress.
