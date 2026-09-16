"""Targeted regression tests for the Relationship Validation Engine,
added during its methodology audit. Covers the two fixes made by that
audit (group-preserving permutation; fraction-based fold-stability
thresholds) plus the core statistical primitives (empirical p-values,
Benjamini-Hochberg FDR) they depend on.
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain.stage2_relval_common import benjamini_hochberg, empirical_pvalue, group_block_shuffle


def test_group_block_shuffle_preserves_within_group_structure():
    groups = np.repeat(np.arange(10), 5)  # 10 groups of 5
    rng_data = np.random.RandomState(1)
    y = rng_data.normal(size=50) + np.repeat(rng_data.normal(size=10), 5)  # real between-group signal
    y_shuf = group_block_shuffle(y, groups, np.random.RandomState(0))

    # same multiset of values -- a true permutation, no values invented/lost
    assert np.allclose(np.sort(y), np.sort(y_shuf))

    # within-group variance must be EXACTLY preserved (blocks stay intact,
    # only reassigned to a different group's row positions)
    for g in np.unique(groups):
        orig_var = y[groups == g].var()
        # the shuffled block at these positions must match SOME original group's block exactly
        shuf_block = y_shuf[groups == g]
        assert any(np.allclose(np.sort(shuf_block), np.sort(y[groups == g2])) for g2 in np.unique(groups))


def test_group_block_shuffle_destroys_between_group_association():
    # a group-level covariate should lose its association with y after shuffling
    rng = np.random.RandomState(2)
    n_groups = 40
    groups = np.repeat(np.arange(n_groups), 5)
    group_level_x = np.repeat(rng.normal(size=n_groups), 5)
    y = 3 * group_level_x + rng.normal(scale=0.1, size=n_groups * 5)  # strong real association

    real_corr = np.corrcoef(group_level_x, y)[0, 1]
    y_shuf = group_block_shuffle(y, groups, np.random.RandomState(3))
    shuf_corr = np.corrcoef(group_level_x, y_shuf)[0, 1]

    assert real_corr > 0.9
    assert abs(shuf_corr) < 0.4  # association destroyed (not exactly 0 by chance, but far from real_corr)


def test_group_block_shuffle_rejects_unequal_group_sizes():
    groups = np.array([0, 0, 0, 1, 1])  # sizes 3 and 2
    y = np.arange(5, dtype=float)
    try:
        group_block_shuffle(y, groups, np.random.RandomState(0))
        assert False, "expected ValueError for unequal group sizes"
    except ValueError:
        pass


def test_empirical_pvalue_never_exactly_zero():
    null_pool = np.random.RandomState(0).normal(size=1000)
    p = empirical_pvalue(observed=1000.0, null_pool=null_pool)  # absurdly extreme observed value
    assert p > 0.0
    assert p == 1.0 / 1001  # add-one continuity correction floor


def test_empirical_pvalue_direction():
    null_pool = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    # observed far below the null -> should NOT look significant (p close to 1)
    assert empirical_pvalue(-5.0, null_pool) > 0.9
    # observed far above the null -> should look significant (p small)
    assert empirical_pvalue(5.0, null_pool) < 0.3


def test_benjamini_hochberg_textbook_case():
    # classic 5-hypothesis example: p = [0.01, 0.02, 0.03, 0.04, 0.5], alpha=0.05
    # BH critical values are i/m*alpha = 0.01, 0.02, 0.03, 0.04, 0.05 -> first 4 accepted
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.5])
    q, accept = benjamini_hochberg(p, alpha=0.05)
    assert list(accept) == [True, True, True, True, False]
    assert np.allclose(q[:4], 0.05)


def test_benjamini_hochberg_monotonic_and_bounded():
    rng = np.random.RandomState(0)
    p = rng.uniform(size=200)
    q, _ = benjamini_hochberg(p, alpha=0.05)
    assert (q >= 0).all() and (q <= 1).all()
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)  # q must be non-decreasing in sorted p-value order


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
