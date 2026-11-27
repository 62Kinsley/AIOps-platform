---
id: disk-io-stress
title: Disk I/O stress
applies_to: disk
keywords: disk, io, i/o, diskio, iops, storage, write, read, fsync, database, slow query, iowait, throughput
summary: One service's disk I/O jumps; disk-bound work (databases, file writes, logging) slows down and the slowdown spreads to its callers.
---

## Symptoms

- **Metrics.** The faulty service's `diskio` jumps at the onset and stays high. Its latency
  rises, most for operations that read or write disk. CPU may rise slightly, since I/O wait and
  the stressor's own work count. Databases (for example carts-db, orders-db, ts-*-mongo) are the
  most sensitive; application services suffer mainly when they log or persist synchronously.
- **Traces.** Operations that touch storage get slower: database client spans (for example
  `find ts.station`, `XRepository.findById`) or the database-backed service's own spans.
- **Logs.** Slow-query warnings or timeouts in the service or in its callers.

## Confirm

1. `detect_anomalies(metric="diskio")`: a strong increase in one service at the onset.
2. `query_metrics(service=<suspect>)`: diskio and latency rise together.
3. If the suspect is a database, its owning service (`get_service_topology`) should show slower
   repository or database spans in `query_traces`.

## Distinguish from

- **Bursty background I/O.** Many services have sporadic disk bursts (log rotation, compaction,
  checkpoints) before the alert as well. A burst that also occurs in the baseline is not a
  fault; look for a sustained change that starts at the onset.
- **More load.** More requests mean more disk I/O in many services together.
- `diskio` is not recorded for every service in every case. A missing series says nothing;
  judge the services where it exists, and rely on latency and traces elsewhere.

## Typical causes

A disk-heavy batch job or backup, a stress process writing to disk, a slow or full volume, an
inefficient query that scans data.
