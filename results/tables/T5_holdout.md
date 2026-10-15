| group | condition | profile | n_runs | n_scenarios | first_error_acc | first_error_lo | first_error_hi | root_cause_acc | root_cause_lo | root_cause_hi | first_error_drop | root_cause_drop | median_bytes_per_run | planted_values_per_run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| manifested | full | C1+C2+C3+C4+C5+C6 | 27 | 14 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 4.82e+04 | 22.12 |
| manifested | profile | C2x+C3l+C6 | 27 | 14 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1.894e+04 | 2.389 |
| manifested | skeleton | skeleton | 27 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1354 | 0 |
| masked | full | C1+C2+C3+C4+C5+C6 | 44 | 14 | 0.9773 | 0.9285 | 1 | 0.9773 | 0.9285 | 1 | 0 | 0 | 4.82e+04 | 22.12 |
| masked | profile | C2x+C3l+C6 | 44 | 14 | 0.9773 | 0.9285 | 1 | 0.9773 | 0.9285 | 1 | 0 | 0 | 1.894e+04 | 2.389 |
| masked | skeleton | skeleton | 44 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0.9773 | 0.9773 | 1354 | 0 |
