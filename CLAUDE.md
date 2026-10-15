# Which Log Line Would Have Told Me? A Logging Ablation on an Agent Pipeline

Personal research project by Rakesh Reddy Karri. Measures which log categories actually help attribute
failures in a three-agent LLM pipeline, and what each category costs in size, latency and sensitive data.

## Research questions
1. Value: how much does each log category contribute to first-error attribution (which step went wrong
   first) and root-cause attribution (which fault mechanism)?
2. Cost: what does each category cost in trace size, added latency and sensitive content stored?
3. Minimum profile: what is the smallest set of categories that keeps attribution within 5 percentage
   points of full logging, and does it hold on held-back runs?

## Use case
Incident triage for a simulated online shop. Do NOT use telecom as the domain.
- Services: api-gateway, auth, catalog, checkout, payment-db, cache, with a fixed dependency graph.
- Six incident types, each with one true root cause and one correct action:
  db_pool_exhausted -> raise pool limit, restart checkout
  bad_deploy -> roll back deploy
  memory_leak -> restart service, open bug
  cross_zone_latency -> fail over to single zone
  expired_certificate -> renew certificate
  cache_eviction_storm -> warm cache, rate-limit
- Each scenario: alerts, metric time series, deploy/change log, log excerpts, dependency graph, plus
  synthetic sensitive fields (customer emails, client IPs, API keys, order IDs) planted by the generator.
- Seeded generator; every scenario stores its ground truth.

## Pipeline (LangGraph)
Evidence agent -> Handover 1 -> Reasoning agent -> Handover 2 -> Action agent
- Evidence agent tools: get_alerts, get_metrics, get_deploys, get_logs, get_topology (mock APIs over the scenario).
- Outputs are Pydantic models, validated before the next step:
  EvidenceBundle: incident_id, affected_service, time_window, findings[{id, source, value}], evidence_confidence
  Diagnosis: root_cause (one of six), suspected_service, reasoning_summary, confidence, evidence_used[ids]
  ActionPlan: action (one of six), target_service, justification, confidence
- Handovers are explicit graph nodes (not hidden state), so faults can be injected and state logged there.
- Outcome check (deterministic): success only if final root_cause AND action match ground truth.

## Fault injection (at most one per run, via hooks; agent code never changes)
| Variant | Class | Injected at | Effect |
| E1 stale_window | evidence | tool layer | get_metrics returns wrong time window |
| E2 wrong_service | evidence | tool layer | tools return a neighbouring service's data |
| E3 missing_alert | evidence | tool layer | decisive alert removed |
| R1 misleading_prior | reasoning | reasoning prompt | biasing hint added |
| R2 unstable_sampling | reasoning | model settings | high temperature on reasoning call |
| I1 dropped_field | interface | handover 1 | time_window or affected_service removed |
| I2 truncated_findings | interface | handover 1 | only first two findings passed on |
| I3 dropped_target | interface | handover 2 | suspected_service removed |

Labels per failed run:
- first_error_step: evidence_agent | handover_1 | reasoning_agent | handover_2 | action_agent
- root_cause_variant: one of the eight variants
Outcomes: manifested (fault + wrong answer; the attribution dataset), masked (fault + right answer),
natural_failure (no fault + wrong answer; excluded from main metrics, counted), clean_success.

Paired design (counterfactual twins):
- Every scenario first gets a clean run (no fault). Its outcome is clean_success or natural_failure.
- Only scenarios whose clean run succeeded are eligible for fault injection. A scenario whose clean
  run failed is never faulted (its failure is natural, so a fault could not be blamed for it).
- Each faulted run stores `twin_run_id`, the run_id of its scenario's clean run.
- Temperature is 0 and the seed is fixed, so the twin is the counterfactual: same scenario, same
  model, same prompts, only the fault differs. Therefore manifested = faulted run fails while its
  twin succeeded (the fault caused the failure); masked = faulted run still succeeds.
- Exception R2 (unstable_sampling): it changes sampling on purpose, so the faulted run is one
  stochastic draw, not a deterministic counterfactual. Its sampling seed is stored; R2 results are
  labelled manifested/masked by the same rule but reported separately and read as a rate, not a cause.
- Eligibility per incident type (clean successes / clean runs) is reported as a result
  (results/eligibility.csv, shown in T1), because it limits how many faulted runs each type can supply.
Run mix target: 60 clean runs and 30 faulted runs per variant, drawn from eligible scenarios; if a
type has too few eligible scenarios, run fewer and report the shortfall. Adjust after a 20-run pilot.

## Logging
Log everything in every run; ablate afterwards by filtering. Each event is one JSON line:
{run_id, step, span_id, category, timestamp, payload}. Write .jsonl while running, compact to Parquet.
| ID | Category | Content |
| C1 | step_io | object each agent received and returned |
| C2 | tool_calls | tool name, arguments, full result |
| C3 | prompt_shown | exact rendered prompt incl. system text |
| C4 | confidence | token logprobs of answer fields + stated confidence |
| C5 | timings | start/end/duration per step, tool call, model call |
| C6 | handover_state | object sent, object received, schema check, field-level diff |
| skeleton | never removed | run_id, step names, order, final answer |

## Attribution and ablation
- Rule-based checker: deterministic checks against the taxonomy (reference ceiling).
- LLM judge: all-at-once over the filtered trace + taxonomy; returns validated
  {first_error_step, root_cause_variant, reason, confidence}.
- Conditions: full; full minus C1..C6 (six); skeleton only. Add-one pass for rule-based only.
- Metrics: first-error accuracy (5 classes), root-cause accuracy (8 classes), macro-F1, per-class P/R,
  confusion matrices, accuracy drop vs full with 95% paired bootstrap CI (2,000 resamples).
  A drop whose CI crosses zero is reported as "no measurable value".

## Cost and privacy
- Size: bytes per run per category, raw and gzip; median and p95.
- Latency: 30 clean scenarios with logging off, all on, and one category on at a time.
- Sensitive exposure: Microsoft Presidio (+ custom recognizers for API keys and order IDs); report % of
  events and entities per run, and Presidio recall against the planted ground truth.
- Redaction test: C2 and C3 also scored in masked form.

## Minimum profile
70/30 design/holdout split by scenario seed, stratified by variant. Rule-based scores all 64 profiles;
LLM judge scores a greedy shortlist. Rule: smallest profile by trace size with both accuracies within
5 points of full; tie-break on less sensitive content. Write results/profile.json and commit BEFORE
opening the holdout. Then score full, profile and skeleton on the holdout. Report failure honestly.

## Hardware and models (MacBook Pro M2, 8 GB)
- Agents: qwen2.5:3b via Ollama. Judge: qwen2.5:7b, run alone after agents finish;
  fallback judge phi4-mini only. Installed models: qwen2.5:3b, qwen2.5:7b, phi4-mini.
- Measured on this Mac: qwen2.5:3b = 2.5 GB, 100% GPU, about 39 tokens/s generation at num_ctx 8192.
- ALWAYS pass options={"num_ctx": 8192} (judge may use up to 8192, never more) on every Ollama request.
  Never rely on the global default: a 131072 context once made a 3B model take 23 GB and hang the Mac.
- Only one model loaded at a time. Ollama with one request at a time. Before each phase, check
  `ollama ps` shows no other model loaded; every run script calls `agents.guard.preflight(model)`
  (reads /api/ps, exits on any foreign model) and records `guard.check_after_run` after each run;
  unload with keep_alive=0 when switching agent -> judge model.
- Ollama 0.40.4 is installed and VERIFIED to return token logprobs: pass "logprobs": true (and optionally
  "top_logprobs": N) to /api/generate or /api/chat; the response has a "logprobs" list of
  {token, logprob, bytes}. Use this for C4. No mlx-lm needed.
- Measured: prompt processing about 300 tokens/s, generation about 40 tokens/s for qwen2.5:3b.
- Every phase reads the previous phase's files and is resumable; a crash loses at most one run.

## Repository layout
configs/experiment.yaml          seeds, run mix, models, split
src/logablation/
  scenarios/  tools/  agents/  pipeline/  faults/  tracing/  attribution/  analysis/  report/
tests/                           pytest
data/ runs/ attribution/ results/ figures/
Makefile targets: scenarios, run, attribute, analyse, report

## Outputs (generated by `make report`, never edited by hand)
Figures: F1 architecture, F2 fault map, F3 manifested vs masked per variant, F4 ablation drops with CIs
(first error and root cause side by side), F5 value vs cost (size vs drop, sized by exposure),
F6 category x variant recall-drop heatmap, F7 profile frontier (64 profiles), F8 confusion matrices
full vs profile, F9 holdout full/profile/skeleton with CIs. Save PNG and SVG.
Tables: T1 dataset summary, T2 attribution results, T3 category costs, T4 minimum profile, T5 holdout.
Files: data/scenarios/*.json, runs/raw/*.jsonl, runs/traces.parquet, runs/outcomes.parquet,
attribution/predictions.parquet, results/metrics.csv, results/costs.csv, results/eligibility.csv, results/profile.json.

## Working rules
- Python 3.11+, uv or venv, ruff, pytest. Type hints everywhere. Every random choice takes a seed.
- Tests must prove: each fault changes exactly what it should; each ablation filter removes exactly one
  category; metrics match hand-computed examples.
- Build in this order: scenarios -> tools -> agents/schemas -> pipeline -> faults -> tracer -> pilot (20 runs)
  -> full runs -> rule-based checker -> LLM judge -> ablation -> costs -> profile + holdout -> report.
- Commit after each working milestone with a clear message. Never commit model weights or large raw data
  (runs/raw goes in .gitignore; commit results/ and figures/).
- After writing each module, explain its design in plain English so the author can present it.
- Commit messages contain no Co-Authored-By or AI attribution lines.
- Run pytest and check its exit code BEFORE committing; never pipe test output through tail in the
  same command as git commit.
- Never fabricate results. If something fails, report it as a result.
