---
id: memory-pressure
title: Memory pressure (memory hog, leak, OOM)
applies_to: mem
keywords: memory, mem, leak, hog, stress, oom, oomkilled, out of memory, heap, gc, garbage collection, swap, restart, crash, killed
summary: One service's memory jumps or ramps up; it may slow down from garbage collection or be killed and restarted, which callers see as errors.
---

## Symptoms

- **Metrics.** The faulty service's `mem` rises at the onset. A stressor or a large allocation
  makes it jump to a plateau; a leak makes it climb steadily. In garbage-collected runtimes
  (Java, Go, Node.js, C#) CPU often rises with it and latency spikes irregularly.
- **Out of memory.** If memory hits the container limit, the process is killed and restarted:
  metrics gap or reset (memory falls back to its normal level), spans of the service stop and
  then resume, and its logs show startup lines (framework banners, "Started ... in N seconds",
  connection pools initialising). Callers log connection errors during the restart.
- **Logs.** OutOfMemoryError or GC warnings in the faulty service, or only the restart lines.

## Confirm

1. `detect_anomalies(metric="mem")`: a strong `mem` increase whose onset is at or just after
   the alert.
2. `query_metrics(service=<suspect>)`: the shape (jump, ramp, sawtooth from repeated restarts)
   and whether CPU and latency move with it.
3. `search_logs(service=<suspect>)`: new templates after the alert, such as startup lines or
   memory errors; `templates_decreased` shows the work that stopped during a restart.
4. Check that the services with errors or latency call the suspect.

## Distinguish from

- **Normal growth.** Caches and JVM heaps grow slowly in many services; a few percent over the
  window is not an anomaly. Look for the service with the largest relative change.
- **CPU saturation.** If CPU rises first and memory stays flat, see cpu-saturation.
- **Code-level faults** can also crash a service; then exceptions, not memory growth, come
  first (see code-level-fault).

## Typical causes

A memory leak in new code, an unbounded cache or queue, a large request or batch, a memory
stress process, a memory limit set too low.
