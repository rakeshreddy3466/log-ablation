import numpy as np
import pandas as pd

from logablation.analysis.dataset import t1_by_type, variant_rates


def _df() -> pd.DataFrame:
    rows = []
    for i in range(10):
        s = f"s{i}"
        t = "bad_deploy" if i < 5 else "memory_leak"
        rows.append((f"{s}-clean", s, t, None, "clean_success", True))
        rows.append((f"{s}-E1", s, t, "E1", "manifested" if i < 4 else "masked", i >= 4))
        rows.append((f"{s}-P0", s, t, "P0", "placebo", i != 0))
    return pd.DataFrame(rows, columns=["run_id", "scenario_id", "incident_type", "variant",
                                       "outcome", "success"])  # fmt: skip


def test_t1_counts() -> None:
    t = t1_by_type(_df()).set_index("incident_type")
    assert t.loc["bad_deploy", ["clean_runs", "eligible", "faulted_runs", "manifested", "masked"]
                 ].tolist() == [5, 5, 5, 4, 1]  # fmt: skip
    assert t.loc["ALL", "manifested"] == 4


def test_rates_hand_computed() -> None:
    r = variant_rates(_df(), n_boot=200).set_index("variant")
    assert r.loc["E1", "rate"] == 0.4
    assert r.loc["P0", "rate"] == 0.1
    assert r.loc["E1", "ci_lo"] <= 0.4 <= r.loc["E1", "ci_hi"]
    assert not np.isnan(r.loc["E1", "diff_lo"])
