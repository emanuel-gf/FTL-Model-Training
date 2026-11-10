"""Names shared by the notebooks and the scripts (stdlib only, runs everywhere)."""

import re


def run_name(kind: str, weights: str) -> str:
    """Name of a model run: its prediction folder and its FiftyOne field.

    e.g. run_name("owlv2", "google/owlv2-base-patch16-ensemble") -> "owlv2_owlv2_base_patch16_ensemble"
    """
    base = weights.rstrip("/").split("/")[-1]
    return re.sub(r"\W+", "_", f"{kind}_{base}").strip("_").lower()
