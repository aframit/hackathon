"""Two focused re-fitting studies, each moving one parameter's label encoding to
match expert orderings, plus a frustration report.

  Case 1: search for new binnings of `distance_to_object`.
  Case 2: recalculate the scoring of `interaction_with_critical_surfaces`.

For prototyping we simulate the interface (reorder-app): a hidden "true" encoding
of the target parameter orders random subsets of scenarios; we then re-fit only
that encoding and check we recover it. To use real data instead, drop the
simulated loop and feed the app's returned orderings into wf.submit_ordering.

Only the target parameter is trained here; every other weight stays at its known
WHC value. Fitting more at once is just a longer `trainable` list.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from refit import RefitWorkflow, load_scenarios, scoring

# Hidden "true" encodings used to generate the simulated orderings (label -> score).
# Chosen to reorder scenarios noticeably vs. the known WHC values.
HIDDEN_TRUTH = {
    "distance_to_object": [1.0, 2.0, 7.0, 10.0],
    "interaction_with_critical_surfaces": [-0.2, 0.0, 0.2, 0.4, 0.4],
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def bin_order_agreement(param: str, encoding: dict) -> float:
    """Rank agreement between an encoding's bin scores and the hidden truth."""
    labels = scoring.ENCODABLE[param]["labels"]
    return spearman(np.array([encoding[l] for l in labels]), np.array(HIDDEN_TRUTH[param]))


def true_params(param: str) -> dict:
    p = scoring.init_params()
    p[scoring.encoding_key(param)] = jnp.asarray(HIDDEN_TRUTH[param], dtype=jnp.float32)
    return p


def true_scores(scenarios, params) -> np.ndarray:
    x = scoring.assemble_features(scenarios.features, params, scenarios.enc_label_idx)
    return np.asarray(scoring.score(params, x))


def run_study(param: str, n_orderings: int = 60, subset_size: int = 10) -> None:
    key = scoring.encoding_key(param)
    labels = scoring.ENCODABLE[param]["labels"]
    scenarios = load_scenarios()

    tscores = true_scores(scenarios, true_params(param))
    truth_bins = np.array(HIDDEN_TRUTH[param])

    wf = RefitWorkflow(scenarios, orderings_path=None, weights_path=None)

    # simulate the interface: serve subsets that isolate the target parameter, and
    # order each primarily by that parameter (tie-break on overall criticality) -
    # i.e. the expert is judging these near-matched scenarios on the studied param.
    rng = np.random.default_rng(0)
    for _ in range(n_orderings):
        batch = wf.serve_contrast(param, n=subset_size, seed=int(rng.integers(1e9)))
        ids = [b["scenario_id"] for b in batch]
        rows = scenarios.index_of(ids)
        li = scenarios.enc_label_idx[param][rows]
        key_score = truth_bins[li] + 1e-6 * tscores[rows]
        wf.submit_ordering([ids[i] for i in np.argsort(-key_score)])

    result = wf.refit(trainable=[key], epochs=600, lr=0.05)

    known = dict(zip(labels, scoring.ENCODABLE[param]["init"]))
    hidden = dict(zip(labels, HIDDEN_TRUTH[param]))
    fitted = wf.effective_encoding(param)

    print(f"\n{'='*70}\nSTUDY: {param}\n{'='*70}")
    print(f"orderings={len(wf.store)}  constraints={result.n_pairs}  "
          f"train_acc={result.pair_accuracy:.3f}")
    print(f"bin-order agreement vs truth: known={bin_order_agreement(param, known):+.3f}  "
          f"recovered={bin_order_agreement(param, fitted):+.3f}")
    print("\nlabel -> score            known    hidden   recovered")
    for lab in labels:
        print(f"  {lab:<26}{known[lab]:>6.2f} {hidden[lab]:>8.2f} {fitted[lab]:>10.2f}")

    print("\nmost frustrated scenarios (can't match the orderings):")
    for row in wf.worst_offenders(result, n=5):
        print(f"  {row['frustration']:.3f}  {row['hazard'][:60]}")


def main() -> None:
    run_study("distance_to_object")
    run_study("interaction_with_critical_surfaces")


if __name__ == "__main__":
    main()
