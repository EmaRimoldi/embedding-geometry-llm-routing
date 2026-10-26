import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rb import pathological as P


def synthetic():
    rng = np.random.default_rng(0)
    fam = np.repeat([0, 1, 2], 40)                       # 3 families × 40 rows
    centers = rng.normal(size=(3, 8)) * 3
    x = centers[fam] + rng.normal(size=(120, 8)) * 0.3
    x[1] = x[0] + 1e-3                                    # rows 0,1: near-identical text
    rewards = rng.integers(0, 2, size=(120, 4)).astype(np.float32)
    rewards[0] = [1, 0, 0, 0]; rewards[1] = [0, 0, 0, 1]  # different solver sets, different argmax
    costs = np.tile(np.array([1e-3, 2e-3, 3e-3, 4e-3]), (120, 1))
    return x, fam, rewards, costs


def test_unit_rows_norm():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    u = P.unit_rows(x)
    assert np.allclose(np.linalg.norm(u[0]), 1.0) and np.all(np.isfinite(u))


def test_argmax_utility_tie_breaks_by_cost_then_index():
    r = np.array([[1.0, 1.0, 0.0]]); c = np.array([[2e-3, 1e-3, 5e-4]])
    assert P.argmax_utility(r, c, lam=0.0)[0] == 1          # tie on reward → cheaper model
    r = np.array([[1.0, 1.0]]); c = np.array([[1e-3, 1e-3]])
    assert P.argmax_utility(r, c, lam=100.0)[0] == 0        # full tie → first index


def test_candidate_pairs_unique_same_family_and_finds_twin():
    x, fam, r, c = synthetic()
    pairs, cos = P.candidate_pairs(P.unit_rows(x), fam, k=5, cos_min=0.5)
    assert pairs.shape[1] == 2 and np.all(pairs[:, 0] < pairs[:, 1])
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    assert np.all(fam[pairs[:, 0]] == fam[pairs[:, 1]])
    assert (0, 1) in {tuple(p) for p in pairs}
    assert cos.shape == (len(pairs),) and np.all(cos >= 0.5)


def test_classify_pairs_flags_twin_pathological():
    x, fam, r, c = synthetic()
    pairs = np.array([[0, 1], [2, 3]])
    am = {100.0: P.argmax_utility(r, c, 100.0)}
    S = P.solver_sets(r); jac = P.jaccard_pairs(S, pairs)
    flags = P.classify_pairs(pairs, am, jac)
    assert flags["pathological_argmax"][0] and flags["pathological"][0] and not flags["benign"][0]
    assert set(flags) == {"pathological_argmax", "pathological_jaccard", "pathological", "benign"}


def test_random_pairs_same_family_and_distinct():
    fam = np.repeat([0, 1], 50); rng = np.random.default_rng(1)
    pr = P.random_pairs(fam, 200, rng)
    assert pr.shape == (200, 2) and np.all(fam[pr[:, 0]] == fam[pr[:, 1]]) and np.all(pr[:, 0] != pr[:, 1])


def test_multiplier_is_ratio_of_normalised_distances():
    m = P.multiplier(np.array([0.5]), np.array([2.5]), 1.0, 1.0)
    assert np.isclose(m[0], 5.0)
    m = P.multiplier(np.array([0.5]), np.array([2.5]), 0.5, 2.5)   # scale change only → 1
    assert np.isclose(m[0], 1.0)


def test_knn_purity_is_one_when_labels_follow_clusters():
    x, fam, r, c = synthetic()
    labels = fam.copy()
    assert P.knn_label_purity(P.unit_rows(x), labels, fam, k=5) == 1.0


def test_clustering_scores_keys_and_degenerate():
    x, fam, r, c = synthetic()
    s = P.clustering_scores(P.unit_rows(x), fam)
    assert set(s) == {"ari", "nmi", "silhouette", "k"} and s["k"] == 3 and s["ari"] > 0.5
    s1 = P.clustering_scores(P.unit_rows(x), np.zeros(120, int))
    assert s1["k"] == 1 and np.isnan(s1["ari"])


def test_bootstrap_mean_diff_contains_true_effect():
    rng = np.random.default_rng(2); a = rng.normal(1.0, 0.1, 500); b = rng.normal(0.0, 0.1, 500)
    d, lo, hi = P.bootstrap_mean_diff(a, b, n_boot=300, seed=0)
    assert lo < 1.0 < hi and abs(d - 1.0) < 0.05


def test_random_pairs_skips_singleton_family():
    fam = np.array([0] * 30 + [1] + [2] * 30); rng = np.random.default_rng(3)
    pr = P.random_pairs(fam, 300, rng)
    assert pr.shape == (300, 2) and np.all(fam[pr[:, 0]] == fam[pr[:, 1]]) and np.all(pr[:, 0] != pr[:, 1])
    assert not np.any(fam[pr[:, 0]] == 1)


def test_pair_cosine_distance_multiple_batches_match_direct():
    """Forcing >1 batch (50 pairs of 8-d vectors, batch=7) must equal the direct formula,
    and the default batch=None (dimension-aware) must agree with it."""
    rng = np.random.default_rng(7)
    x = P.unit_rows(rng.normal(size=(60, 8)))
    pairs = np.stack([rng.integers(0, 60, 50), rng.integers(0, 60, 50)], axis=1)
    direct = np.clip(1.0 - np.sum(x[pairs[:, 0]] * x[pairs[:, 1]], axis=1), 0.0, 2.0)
    batched = P.pair_cosine_distance(x, pairs, batch=7)          # 50 / 7 -> 8 batches
    assert np.allclose(batched, direct, atol=1e-6)
    assert np.allclose(P.pair_cosine_distance(x, pairs), direct, atol=1e-6)   # batch=None


def test_bootstrap_ratio_of_medians_by_query_brackets_truth():
    """Two synthetic pair groups with medians 2 and 1: estimate ~ 2, interval brackets 2."""
    rng = np.random.default_rng(5); n_rows = 300
    pp = rng.integers(0, n_rows, size=(400, 2)); pb = rng.integers(0, n_rows, size=(400, 2))
    mp = rng.normal(2.0, 0.10, 400); mb = rng.normal(1.0, 0.05, 400)
    est, lo, hi = P.bootstrap_ratio_of_medians_by_query(mp, pp, mb, pb, n_rows, n_boot=200, seed=0)
    assert abs(est - 2.0) < 0.1 and lo < 2.0 < hi and lo < est < hi


def test_weighted_median_matches_repetition():
    v = np.array([1.0, 2.0, 3.0]); w = np.array([1.0, 5.0, 1.0])
    assert P.weighted_median(v, w) == 2.0
    assert np.isnan(P.weighted_median(v, np.zeros(3)))


def test_argmax_utility_respects_availability_mask():
    r = np.array([[0.0, 1.0, 0.5]]); c = np.array([[1e-3, 2e-3, 3e-3]])
    assert P.argmax_utility(r, c, lam=0.0)[0] == 1                     # no mask -> best overall
    av = np.array([[True, False, True]])
    assert P.argmax_utility(r, c, lam=0.0, available=av)[0] == 2       # model 1 not run


def test_unit_euclidean_matches_direct():
    rng = np.random.default_rng(5); x = P.unit_rows(rng.normal(size=(20, 6)))
    pairs = np.array([[0, 1], [2, 3], [4, 4]])
    cos = P.pair_cosine(x, pairs)
    direct = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    assert np.allclose(P.unit_euclidean(cos[:2]), direct[:2], atol=1e-5)
    assert np.allclose(P.pair_unit_euclidean(x, pairs), direct, atol=1e-6)


def test_neighbor_rank_small_example():
    x = P.unit_rows(np.array([[1, 0], [1, 0.1], [1, 0.5], [0, 1], [1, 0.05]], dtype=float))
    fam = np.array([0, 0, 0, 0, 1])
    pairs = np.array([[0, 1], [0, 2]])
    r_ij, r_ji, n_f = P.neighbor_rank(x, fam, pairs)
    assert list(r_ij) == [1, 2] and list(n_f) == [4, 4]      # row 4 is another family: excluded
    assert r_ji[0] == 1 and r_ji[1] == 2                      # 2's neighbours: 1 (closer), then 0


def test_family_percentile_extremes():
    rng = np.random.default_rng(0); x = P.unit_rows(rng.normal(size=(200, 5)))
    fam = np.zeros(200, int); pairs = np.array([[0, 1]])
    assert P.family_percentile(x, fam, pairs, np.array([1.0]), rng)[0] == 0.0
    assert P.family_percentile(x, fam, pairs, np.array([-1.0]), rng)[0] == 1.0
