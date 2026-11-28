# AIOps-platform

Root cause analysis (RCA) for microservice systems, built on the
[RCAEval](https://github.com/phamquiluan/RCAEval) benchmark (WWW 2025): 735 failure
cases from Online Boutique, Sock Shop and Train Ticket, with metrics, logs and traces.

**Phase 1 — RCAEval data pipeline:** download the cases, read metrics, logs and
traces into pandas, and isolate the ground truth so an RCA method can't see the answer.

**Phase 2 — MCP servers:** a Monitoring server and a Knowledge server give an LLM agent
seven read-only tools to investigate one case ([below](#phase-2--mcp-servers)).

**Evaluation:** `aiops-eval` scores any RCA method the way RCAEval does, with a fixed
dev/test split and two baselines that need no LLM ([below](#evaluation-and-baselines)).

## Quickstart

```bash
uv sync
uv run aiops-data download --all          # 735 cases, 3.4 GB (or: -d RE2-OB, --max-reps 1, ...)
uv run aiops-data build                   # blind views + ground-truth tables + leakage audit
uv run aiops-data inspect                 # what is downloaded / built
uv run aiops-data inspect RE2-OB-0078b4ee # load one case's metrics, logs and traces
uv run aiops-monitoring-server            # MCP servers (stdio); see Phase 2 below
uv run aiops-knowledge-server
uv run aiops-eval run anomaly             # score the no-LLM baseline on every case
uv run pytest
```

Data goes to `./data` (or `$AIOPS_DATA_DIR`). Downloads come from the Parquet copy on
[Hugging Face](https://huggingface.co/datasets/phamquiluan/RCAEval), pinned to one
dataset commit. They resume where they stopped and skip files already on disk. `build`
rebuilds a dataset from scratch (all 735 cases take a few seconds) and keeps its case ids.

## Data layout

The pipeline keeps three zones apart:

```
data/
├── raw/rcaeval/     LABEL-BEARING  Hugging Face mirror; directory names are the answers
│   ├── cases.parquet                 (re2ob_checkoutservice_cpu_1 = service + fault)
│   └── re2ob_checkoutservice_cpu_1/{metrics,logs,traces}.parquet, inject_time.txt
├── labels/          LABEL-BEARING  for evaluators only
│   ├── RE2-OB.parquet                ground truth, one row per case_id
│   └── RE2-OB.json                   salt, source revision, leakage audit
└── blind/           LABEL-FREE     the only zone an RCA method reads
    └── RE2-OB/
        ├── manifest.json             case ids, candidate services
        └── cases/RE2-OB-0078b4ee/
            ├── metrics.parquet       byte-identical copies of the raw telemetry
            ├── logs.parquet
            ├── traces.parquet
            └── case.json             inject time, time range, files, row counts
```

## Using it from code

An RCA method sees only the blind view:

```python
from aiops.data import BlindDataset

for case in BlindDataset.open("RE2-OB"):
    metrics = case.metrics                    # index: time (unix s); one column per {service}_{metric}
    normal = metrics[metrics.index < case.inject_time]
    faulty = metrics[metrics.index >= case.inject_time]
    logs, traces = case.logs, case.traces     # None when the system records none
    span_p90 = traces.groupby("service")["duration_us"].quantile(0.9)
```

An evaluator joins predictions to the labels by `case_id`:

```python
from aiops.data.ground_truth import load_labels

labels = load_labels("data/labels", "RE2-OB")   # indexed by case_id
labels.loc["RE2-OB-0078b4ee", ["root_cause_service", "fault", "indicator"]]
```

## Telemetry readers

`aiops.data.readers` normalises each source. All three share one clock: `time` is
int64 unix seconds, the unit of the metrics and of the inject time.

| Source | Frame | Notes |
|---|---|---|
| `read_metrics` | index `time`; float64 columns `{service}_{metric}` | ~1 sample/s. Nulls mean "not observed" and are kept; `fill=True` forward-fills, back-fills, then zero-fills (RCAEval's convention). `metrics_long()` gives `time, service, metric, value`. |
| `read_logs` | `time, service, message` | Sorted by time. A missing message becomes `""`, so log volume stays intact. |
| `read_traces` | `time, start_time_us, duration_us, trace_id, span_id, parent_span_id, service, operation, method, status_code` | Sorted by start time. Root spans have a null parent. `status_code` is nullable: Train Ticket records none, Online Boutique uses gRPC codes (0 = OK). |

Service names match across sources. RE2-OB traces call the frontend `frontendservice`, and
the reader maps that to `frontend`, its name in the metrics and logs. A scan of all 735
cases found no other mismatch. Sock Shop records no traces, RE1 has metrics only, and
`re2tt_ts-auth-service_cpu_1` has no logs.

## Ground-truth isolation

A raw case gives away its answer in several ways. `aiops-data build` closes each one:

| Leak | How the blind view handles it |
|---|---|
| Directory name `re2ob_checkoutservice_cpu_1` | Renamed to `RE2-OB-<8 hex>`, a prefix of HMAC-SHA256(salt, name). The salt is random, stored only in `labels/`, and reused on rebuild, so ids stay stable. A plain hash of the public case names could be recomputed. |
| `cases.parquet`, `root_cause.txt` (the root-cause log line in 8 RE3-SS cases) | Not copied. Their content goes to `labels/`. |
| `inject_time.txt` | Its value moves to `case.json`. RCAEval gives the inject time to RCA methods as the anomaly-detection time, so it is an input. Note that it is always the midpoint of the window. |
| File inode numbers and timestamps | Telemetry is **copied**, in case-id order. A hardlink, which RCAEval's own export uses, keeps the raw file's inode and mtime. On the mirror both follow the alphabetical order of the answer-bearing names (rank correlation 0.90–1.00), so `ls -li` would undo the renaming. `--hardlink` saves the 3.2 GB but the audit flags it. |

Every build audits its own output before swapping it in. It aborts if a file other than
telemetry or `case.json` is present, or if a raw case name appears in any path, JSON file
or Parquet metadata. It also reports two properties of the benchmark data itself that a
method could exploit (⚠ = flagged):

| Dataset | Cases | Column-presence cheat AC@1 (random) | Back-to-back cases with same label (chance) | Metric columns common to all cases |
|---|---:|---:|---:|---:|
| RE1-OB | 125 | 0.10 (0.05) ⚠ | 61% (3%) ⚠ | 24 |
| RE1-SS | 125 | 0.22 (0.06) ⚠ | 57% (3%) ⚠ | 57 |
| RE1-TT | 125 | 0.10 (0.02) ⚠ | 15% (3%) ⚠ | 195 |
| RE2-OB | 90 | 0.15 (0.06) ⚠ | 33% (2%) ⚠ | 67 |
| RE2-SS | 90 | **0.68** (0.06) ⚠ | 33% (2%) ⚠ | 73 |
| RE2-TT | 90 | 0.04 (0.01) | 31% (2%) ⚠ | 337 |
| RE3-OB | 30 | 0.07 (0.04) | 0% (7%) | 68 |
| RE3-SS | 30 | 0.12 (0.04) ⚠ | 14% (8%) | 80 |
| RE3-TT | 30 | 0.05 (0.01) | 31% (10%) ⚠ | 291 |

- **Column presence.** The set of metric columns differs between cases, and which columns
  a case has partly reveals its root cause. In RE2-SS, `carts_diskio` appears in 16 of
  the 18 carts-fault cases but only 12 of the other 72. It holds values from before the
  injection too, so this is how the data was collected, not an effect of the fault. A
  cheat that never reads a value finds 68% of RE2-SS root causes this way. The telemetry
  is left unchanged so that results stay comparable with published RCAEval numbers. To
  remove the shortcut, restrict a method to `BlindDataset.common_metric_columns()`. This
  costs columns, notably in RE1-OB (24 of 95 remain, since it mixes two metric naming
  schemes), and most of the `{service}_diskio` columns that RCAEval scores RE2-OB and
  RE2-SS disk faults against.
- **Injection order.** Cases were injected in batches, so cases close in absolute time
  often share a root cause and fault. Evaluate one case at a time, and don't give a
  method other cases' inject times. Shifting the timestamps would not hide this, because
  log messages embed them.

The full audit for each dataset is in `data/labels/<DATASET>.json`.

## Phase 2 — MCP servers

Two [MCP](https://modelcontextprotocol.io) servers (Python SDK 2.x) give an agent what an
on-call engineer would use: telemetry of the incident, and knowledge about the system.

| Server | Tool | Answers |
|---|---|---|
| Monitoring | `get_incident_overview` | System, alert time, window, which telemetry exists, services and their metrics, root-cause candidates |
| | `detect_anomalies` | Metric series ranked by robust z-score against their own pre-alert baseline, with onset times and a per-service summary |
| | `query_metrics` | Downsampled time series of a service's metrics, or of one metric across services |
| | `search_logs` | Log templates whose frequency changed after the alert (new errors, restarts, services that went quiet); regex search |
| | `query_traces` | Latency and error rate per operation and per service-to-service call, caller side vs callee side; one trace as a call tree |
| Knowledge | `get_service_topology` | Services, who calls whom (with evidence), transitive dependencies and dependents, known background log noise |
| | `search_runbooks` | BM25 search over 9 runbooks: triage workflow, fault propagation, and one per fault type (CPU, memory, disk, delay, loss, socket, code) |

```bash
uv run aiops-monitoring-server                    # stdio, reads $AIOPS_DATA_DIR (or --data-dir)
uv run aiops-monitoring-server --common-metrics   # hide the column-presence shortcut
uv run aiops-knowledge-server
uv run aiops-monitoring-server --transport streamable-http --port 8001   # or over HTTP

# Try the tools in Claude Code, or in the MCP Inspector web UI:
claude mcp add aiops-monitoring -e AIOPS_DATA_DIR=$HOME/aiops-data -- uv run --project "$PWD" aiops-monitoring-server
claude mcp add aiops-knowledge -- uv run --project "$PWD" aiops-knowledge-server
npx @modelcontextprotocol/inspector uv run aiops-monitoring-server
```

The tool logic is plain Python, usable without MCP:

```python
from aiops.monitoring import MonitoringService
from aiops.knowledge import KnowledgeBase

MonitoringService().detect_anomalies("RE2-OB-0078b4ee", top_k=5)
KnowledgeBase().service_topology("Online Boutique", "cartservice")
```

**Conventions.** Every monitoring tool takes a case id. Times in requests and results are
seconds relative to the alert (RCAEval's inject time), negative before it; the part of a
window before the alert is the baseline. Results are compact JSON, rounded to 4 significant
digits and bounded (top-k lists, at most 30 series per call): with the default limits a call
costs at most about 3k tokens, and on the real data every tool answers in under 1.5 s. A bad argument (unknown case,
service or metric, empty window) comes back as a tool error that names the valid choices.

**Isolation.** The Monitoring server opens cases only through `BlindDataset`. No tool lists
case ids: other cases' inject times would reveal labels through the injection order (see
above). With `--common-metrics` it exposes only the metric columns that every case of the
dataset has. The Knowledge server reads no case data at all. Its topology comes from each
system's published architecture and, for Train Ticket, from the calls seen in the pre-alert
traces of all RE2-TT and RE3-TT cases; every call records its evidence (`traces`,
`documented`, `db-client-spans`, `naming`, `gateway`). Its runbooks describe fault types in
general terms. A test checks that no knowledge file names a case, a case id or a label.

**Tests.** `tests/test_monitoring.py` runs the five monitoring tools on a synthetic case with
a planted incident (cartservice CPU +20, slow spans, a new error log). `tests/test_knowledge.py`
covers topology queries and runbook ranking. `tests/test_mcp_servers.py` calls all seven tools
through an MCP client, in process and over stdio. `tests/test_mcp_real_data.py` runs them on
one real case per dataset, checks the size budget, and checks the topology against the
traces (skipped without data).

## Evaluation and baselines

`aiops-eval` runs an RCA method over the blind view, one case at a time, and scores it
the way RCAEval does.

```bash
uv run aiops-eval run anomaly                            # every built dataset, 735 cases (~15 s)
uv run aiops-eval run anomaly -d RE2-OB --split dev --limit 5
uv run aiops-eval score anomaly --split test --by-fault  # re-score a finished run
```

**Scores.** Per dataset, AC@k is the share of cases whose root cause is in the method's
top k, and Avg@5 is the mean of AC@1 to AC@5 (RCAEval `benchmark/evaluation.py`). Service
names follow RCAEval's rules: a ranked name counts as the service before its first
underscore, with `-db` removed, and only a service's first appearance counts. `fine` is the
metric level: the answer is the root-cause service plus its fault's indicator (cpu, mem,
socket, diskio, or latency for delay and loss), and `latency-90` counts as `latency`. RE3's
code-level faults have no indicator, so they are scored at the service level only.
`random AC@1` is what a random ranking of the candidate services would score.

**Splits.** `--split dev` takes one case per (root-cause service, fault) group, the one
with the smallest id, and `test` is the rest (543 of the 735 cases). Tune a method on dev
and report test. The evaluator makes the split from the labels; a method only ever sees
the case in front of it.

**Runs.** A run lives in `$AIOPS_DATA_DIR/runs/<name>/`. `predictions.jsonl` gets one line
per case as it finishes, so an interrupted run resumes. A case whose method raised counts
as a miss and runs again next time. Scoring adds `scores.json` (summaries for all cases, dev
and test) and `case_scores.csv`, which contain the ground truth, so keep agents away from
`runs/` as from `labels/`. A method has
a `name`, `params` and `rank(case) -> Prediction`. The runner and the baselines never
import the labels module, and a test checks that.

**Baselines** (test split, 543 cases). `anomaly` ranks services by their most anomalous
metric series, using the robust z-score behind `detect_anomalies`. `anomaly-common` does
the same on the metric columns that every case of the dataset has.

| Dataset | `anomaly` AC@1 | AC@3 | Avg@5 | fine Avg@5 | `anomaly-common` AC@1 | Avg@5 | random AC@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RE1-OB | 0.88 | 0.94 | 0.93 | 0.85 | 0.72 | 0.80 | 0.05 |
| RE1-SS | 0.95 | 0.99 | 0.98 | 0.79 | 0.96 | 0.99 | 0.08 |
| RE1-TT | 0.73 | 0.91 | 0.89 | 0.59 | 0.76 | 0.90 | 0.02 |
| RE2-OB | 0.85 | 0.97 | 0.95 | 0.74 | 0.90 | 0.97 | 0.06 |
| RE2-SS | 0.87 | 0.98 | 0.96 | 0.75 | 0.97 | 0.99 | 0.08 |
| RE2-TT | 0.85 | 0.95 | 0.92 | 0.61 | 0.83 | 0.92 | 0.01 |
| RE3-OB | 0.60 | 0.80 | 0.76 | – | 0.60 | 0.76 | 0.04 |
| RE3-SS | 0.48 | 0.95 | 0.85 | – | 0.43 | 0.84 | 0.05 |
| RE3-TT | 0.68 | 1.00 | 0.90 | – | 0.77 | 0.95 | 0.01 |
| Mean | 0.76 | 0.94 | 0.90 | 0.72 | 0.77 | 0.90 | 0.05 |

What the numbers say:

- **Resource faults are easy for a metric scorer.** On RE1 and RE2 the stressed
  container's own CPU, memory, disk or socket series stands out. Over all 90 RE2-TT cases,
  the baseline's service-level Avg@5 per fault (cpu 0.99, mem 1.00, disk 1.00, socket
  0.99, delay 0.95, loss 0.65) matches or beats the BARO results in RCAEval's README
  (0.72, 0.99, 1.0, 0.83, 0.63, 0.64).
- **It is not a leak.** Scored against another case's answer, the same predictions get
  AC@1 0.14–0.34, about one over the number of root-cause services in the dataset.
- **It doesn't rely on column presence.** Restricting it to the common columns leaves the
  mean unchanged. RE1-OB drops (only 24 of its 95 columns are common) and RE2-SS gains.
- **Its weak spots are where an agent can help.** These are network loss (RE2-TT loss
  Avg@5 0.65), RE3's code-level faults (AC@1 0.43–0.77) and fault propagation.
  Error-count series that were zero before the alert get the capped score of 1000, so
  callers that start logging errors outrank the faulty service. With RCAEval's ±10-minute
  windows, which leave more series at zero before the alert, RE3-TT AC@1 falls from 0.67
  to 0.07. The baseline uses the whole window (12–30 minutes, depending on the dataset).
- **The metric level is harder** (mean fine Avg@5 0.72). The right service is often found
  through another metric than its fault's indicator. Over all RE2-TT cases, the fine
  Avg@5 is 0.17 for memory faults and 0.29 for socket faults.

**Tests.** `tests/test_evaluation.py` covers RCAEval's scoring rules, the split, the
runner's resume and error handling, label isolation of the method side, and the command
line on the synthetic mirror.

## Troubleshooting

**macOS with `~/Desktop` or `~/Documents` synced to iCloud Drive.** Two symptoms:

- `ModuleNotFoundError: No module named 'aiops'` a few minutes after `uv sync`. macOS marks
  the files in `.venv` hidden, and Python 3.12+ skips hidden `.pth` files, which breaks the
  editable install. `chflags -R nohidden .venv` fixes it until the flag comes back. Tests
  are unaffected, because pytest imports from `src/` directly.
- `Parquet magic bytes not found`, or reads that hang. When the disk runs low, iCloud
  evicts files to cloud-only placeholders (`ls -lO` shows `dataless`): the data, but also
  `.venv` and source files.

Keep the data outside iCloud (`export AIOPS_DATA_DIR=~/aiops-data`), and preferably the
project or at least its environment too (`export UV_PROJECT_ENVIRONMENT=~/.venvs/aiops-platform`).
