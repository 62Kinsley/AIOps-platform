---
id: triage-workflow
title: Root cause triage workflow
applies_to: any incident
keywords: start, triage, workflow, investigate, root cause, rca, procedure, checklist, steps, hypothesis, evidence, report
summary: The order in which to use the monitoring and knowledge tools to find the root-cause service of an incident, and how to justify the answer.
---

## Goal

Name the root-cause service: the service where the fault was introduced. It is one of the
`root_cause_candidates` in `get_incident_overview`. If the evidence allows, also name the fault
type (CPU, memory, disk I/O, network delay, packet loss, socket exhaustion, or a code-level
fault) and give a ranked list of the next most likely candidates.

## Steps

1. **Scope.** Call `get_incident_overview(case_id)`. Note the system, which telemetry exists
   (some cases have metrics only), and the window around the alert. Times in every monitoring
   tool are seconds relative to the alert.
2. **Find what changed.** Call `detect_anomalies(case_id)`. Read the per-service summary first:
   which services have strong anomalies (score >= 10), in which metrics, and when they began
   (`onset_s`).
3. **Place the anomalies on the call graph.** Call `get_service_topology(system)`. Symptoms
   travel from a faulty service to its callers (see the fault-propagation runbook), so the
   cause is usually the anomalous service furthest from the entry point whose anomaly its own
   dependencies cannot explain.
4. **Form hypotheses.** For the two or three best candidates, decide which fault type would
   produce their pattern of anomalies. Use `search_runbooks` with the symptoms you see, e.g.
   "cpu up latency up callers slow" or "new exceptions in logs, metrics normal".
5. **Test each hypothesis** with the drill-down tools:
   - `query_metrics(service=...)`: does the resource anomaly start at the onset and persist?
   - `query_traces(service=...)` (Online Boutique, Train Ticket): is the suspect's own work
     slower (callee side), or is time lost on the way to it (caller side only)?
   - `search_logs(service=...)`: new error templates, restarts, or a service that went quiet.
6. **Rule out victims.** For each other anomalous service, explain its anomaly as an effect of
   the suspect (it calls the suspect, its onset is the same or later, its own resources are
   normal). An anomaly you cannot explain is a reason to reconsider.
7. **Report** the root-cause service, the fault type if clear, the evidence (metric, log or
   trace observations with times), and the alternatives you ruled out and why.

## Pitfalls

- The entry point (frontend, front-end, ts-ui-dashboard) is anomalous in almost every incident,
  because every slowdown reaches it. It is rarely the cause.
- A service whose traffic drops shows lower CPU, workload and log volume. That is a victim
  pattern: its callers send less, or a dependency fails first.
- Which metrics or log lines exist for a service is not evidence; judge values and changes.
- Background errors that exist before the alert are noise; `get_service_topology` lists the
  known ones per system.
