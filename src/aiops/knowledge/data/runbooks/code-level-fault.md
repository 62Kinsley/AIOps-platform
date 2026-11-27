---
id: code-level-fault
title: Code-level fault (application bug)
applies_to: f1, f2, f3, f4, f5 (RCAEval code-level faults)
keywords: code, bug, exception, error, stack trace, traceback, 500, crash, wrong result, regression, deploy, logic, null pointer, attribute error, type error, unhandled, missing call, status code
summary: A bug in one service's code makes it throw errors or behave differently while resource metrics stay normal; logs and trace status codes show where.
---

## Symptoms

- **Logs.** This is where code faults show most clearly. The faulty service emits new error
  templates after the alert: exceptions and stack traces (NullPointerException, AttributeError,
  TypeError), HTTP 500 responses, "failed to ..." messages. Callers log failures that name it
  ("rpc error ... Unavailable", "500 from ...", "upstream connect error"). Some bugs change
  behaviour without errors: normal messages disappear (`templates_decreased`) or new
  non-error messages appear.
- **Traces.** Where status codes exist (Online Boutique), the faulty service's spans, and the
  callers' spans of calls to it, carry error codes. Call patterns can change: calls that stopped
  after the alert (a missing call) or new ones.
- **Metrics.** Often unremarkable. Error rates and latency of the faulty service and its
  callers may rise; a crashing service shows restarts (metric gaps, memory dropping back to its
  startup level).

## Confirm

1. `search_logs()` without a query: look at `services` for the one with new templates and new
   error-like lines after the alert, and at `templates_increased` for the first new exception
   (`first_after_s`).
2. `search_logs(query="exception|error|traceback")`: read the raw `examples`. The service that
   throws the exception is a stronger candidate than services that only report failed calls
   to it.
3. `query_traces()`: error rates and `operations_new_after_alert` /
   `operations_stopped_after_alert` for the suspect (Online Boutique, Train Ticket).
4. `get_service_topology`: callers that log failures should call the suspect.

## Distinguish from

- **Background errors.** Several services log errors constantly, before the alert as well
  (`get_service_topology` lists the known ones). Only a change matters: a new template or a
  higher rate.
- **Errors caused by a resource fault elsewhere.** Timeouts and connection errors in a service
  often come from a slow or failing dependency. A code fault produces errors at the service
  itself, usually without a resource anomaly in its dependencies.

## Typical causes

A faulty deployment or configuration, an unhandled exception path, wrong parameters or return
values, a missing call or check, an incompatible change between two services.
