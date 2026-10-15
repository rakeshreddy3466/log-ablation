# Judge development log

Dev set: `results/judge_dev_set.json` (design side only, 2 runs per variant, fixed seed).
Degenerate = one predicted label above 60% of predictions, or root-cause accuracy below 30%.
Chance for root cause is 12.5% (1 of 8). No prompt change was made on the holdout side.

## v1 on qwen2.5:7b (2026-10-16, sanity sample of 10 design-side manifested runs, not the dev set)

- first-error accuracy: 50% (5/10)
- root-cause accuracy: 20% (2/10); rule checker on the same 10 runs: 90% / 90%
- predicted label distribution: {'E2': 10}
- most common label: E2 at 100% of predictions
- valid answers: 10/10; about 21 s/call; max prompt tokens 3,871
- degenerate: True. Output schema was first_error_step, root_cause_variant, reason, confidence;
  variants listed in fixed order. This sample was drawn before the dev set existed.
