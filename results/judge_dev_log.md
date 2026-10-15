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

## v2 on qwen2.5:7b (2026-10-16, dev set, full)

- first-error accuracy: 31% (16 runs)
- root-cause accuracy: 12% (chance 12.5%)
- predicted label distribution: {'E3': 4, 'E1': 3, 'R1': 2, 'E2': 2, 'I3': 1, 'I1': 1, 'R2': 1, 'invalid': 1, 'I2': 1}
- most common label: E3 at 25% of predictions
- correct per true variant: {'E1': 1, 'E2': 0, 'E3': 0, 'R1': 0, 'R2': 1, 'I1': 0, 'I2': 0, 'I3': 0}
- valid answers: 15/16; 28.0 s/call; max prompt tokens 4095
- degenerate (one label > 60% or root-cause < 30%): True

## v3 on qwen2.5:7b (2026-10-16, dev set, full)

- first-error accuracy: 44% (16 runs)
- root-cause accuracy: 19% (chance 12.5%)
- predicted label distribution: {'I2': 4, 'E1': 4, 'E2': 2, 'I1': 2, 'I3': 1, 'E3': 1, 'R1': 1, 'R2': 1}
- most common label: I2 at 25% of predictions
- correct per true variant: {'E1': 0, 'E2': 0, 'E3': 0, 'R1': 1, 'R2': 1, 'I1': 1, 'I2': 0, 'I3': 0}
- valid answers: 16/16; 43.4 s/call; max prompt tokens 3871
- degenerate (one label > 60% or root-cause < 30%): True

## v3 on phi4-mini:latest (2026-10-16, dev set, full)

- first-error accuracy: 25% (16 runs)
- root-cause accuracy: 12% (chance 12.5%)
- predicted label distribution: {'I2': 7, 'I1': 5, 'invalid': 2, 'E1': 1, 'I3': 1}
- most common label: I2 at 44% of predictions
- correct per true variant: {'E1': 0, 'E2': 0, 'E3': 0, 'R1': 0, 'R2': 0, 'I1': 0, 'I2': 1, 'I3': 1}
- valid answers: 14/16; 17.2 s/call; max prompt tokens 3468
- degenerate (one label > 60% or root-cause < 30%): True

## Decision

- Versions tried: v1, v2, v3 (v3 allowed because v2 was still degenerate; no further versions).
- Best on the dev set: v3 on qwen2.5:7b, first error 44%, root cause 19% (3/16, chance 12.5%).
- phi4-mini with v3 (comparison only): first error 25%, root cause 12.5%, 2 invalid answers.
- v3 is frozen as the final version. It is still degenerate by the pre-set rule (root cause
  under 30%), so the overnight run was NOT started. The judge's failure is a result:
  with budgeted, generically shortened traces a 7B judge does not find these faults, while the
  rule checker reaches about 92% on the same kind of runs.
- Caveats: 16 dev runs (one variant has a masked run in the set), so 3/16 vs chance is not a
  statistically firm statement either way; the dev runs are also part of the design-side set
  that the judge would later be scored on.
