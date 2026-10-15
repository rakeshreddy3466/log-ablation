"""Save the design-side predictions of full / frozen profile / skeleton (for F8).

Pure recomputation with the frozen profile; the numbers equal those already in
results/rules_ablation.csv and results/profile_candidates_manifested.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..attribution.ablation import load_side
from .privacy import Detector
from .profile import FULL, Choice, predict_profiles


def main() -> None:
    prof = json.loads(Path("results/profile.json").read_text())
    chosen: Choice = {c: None for c in FULL} | prof["categories"]
    skeleton: Choice = {c: None for c in FULL}
    pred = predict_profiles(
        load_side("design"), [FULL, chosen, skeleton], Detector(False), Detector(True)
    )
    Path("attribution").mkdir(exist_ok=True)
    pred.to_parquet("attribution/predictions_profile_design.parquet", index=False)
    print(f"{len(pred)} predictions written")


if __name__ == "__main__":
    main()
