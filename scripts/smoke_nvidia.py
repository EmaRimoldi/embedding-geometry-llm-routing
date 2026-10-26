#!/usr/bin/env python3
"""CPU/CUDA smoke of the final utility implementation on explicitly synthetic inputs."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    scratch_base = ROOT / "results/reproduced/reviewer_nvidia"
    scratch_base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=scratch_base))
    cache = work / ".cache"
    cache.mkdir()
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", XDG_CACHE_HOME=str(cache),
                      MPLCONFIGDIR=str(cache / "matplotlib"), TMPDIR=str(cache),
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    sys.path.insert(0, str(ROOT / "src/nvidia_utility"))
    import common_u as common
    import probe
    import train

    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable. Use --device cpu or an allocated GPU.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(123)
    panel, split = "routerbench-0shot", "ood_0"
    n_train, n_test, models, components = 60, 12, 3, 100
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    layers, penalties, seeds = list(range(15, 21)), [0, 2], list(range(42, 52))
    latent = rng.normal(size=(n_train + n_test, 6))
    features = {layer: (latent @ rng.normal(scale=.08, size=(6, components))
                       + rng.normal(scale=.02, size=(len(latent), components))).astype(np.float32)
                for layer in layers}
    data = dict(r=rng.uniform(.1, .9, size=(len(latent), models)),
                c=rng.uniform(.8, 1.2, size=(len(latent), models)),
                fam=np.array([0] * 30 + [1] * 30 + [2] * n_test),
                query_ids=np.array([f"synthetic-{i}" for i in range(len(latent))]))
    config = dict(smoke_only=True, tasks=[[panel, split]], layers=layers,
                  lambdas={panel: penalties}, seeds=seeds, keep_seeds=5)
    config_path = work / "configs/smoke_utility.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    split_path = work / "configs/baseline_splits" / (panel + ".json")
    split_path.parent.mkdir()
    split_path.write_text(json.dumps({split: dict(train=tr.tolist(), test=te.tolist())}))
    # Redirect only this process's data/path adapters to a synthetic experiment.
    # Scientific source and shipped configuration bytes are never edited.
    common.ROOT, common.CONFIG = work, config_path
    common.OUT = work / "results/utility"
    common.base.ROOT = work
    common.cfg.cache_clear()
    common.splits.cache_clear()
    common.sha.cache_clear()
    common.data = lambda requested: data if requested == panel else None
    common.features = lambda requested, fold, layer: features[layer]
    for module in (probe, train):
        module.data = common.data
        module.cfg = common.cfg
        module.features = common.features
    probe.hardware = lambda: dict(scope="synthetic_smoke")

    def verify_synthetic():
        assert json.loads(config_path.read_text()) == config
        assert not np.intersect1d(tr, te).size
        assert not set(data["fam"][tr]) & set(data["fam"][te])
    # Production verify() requires full frozen datasets. Its integrity gate is
    # deliberately outside this toy test; verify_release.py covers package bytes.
    probe.verify = verify_synthetic
    probe.run(0)
    destination = common.dest(panel, split)
    selection_path = destination / "selection.json"
    selected = json.loads(selection_path.read_text())
    folds = probe.get_folds(panel, split)
    for fit_rows, val_rows in folds:
        assert not np.intersect1d(fit_rows, val_rows).size
        assert set(fit_rows) | set(val_rows) == set(range(n_train))
        assert not set(tr[np.r_[fit_rows, val_rows]]) & set(te)
    y = probe.utility_targets(panel, split)
    for li, penalty in enumerate(penalties):
        for model in range(models):
            np.testing.assert_array_equal(y[:, li * models + model],
                data["r"][tr, model] - penalty * data["c"][tr, model])
    score_cache, ridge_checks = {}, 0
    for layer in selected["evaluated_layers"]:
        report = json.loads((destination / "probes" / f"layer_{layer}.json").read_text())
        score_cache[layer] = report["mean_mse"]
        with np.load(destination / "probes" / f"layer_{layer}.npz") as saved:
            for fi, (fit_rows, val_rows) in enumerate(folds):
                x = features[layer][tr].astype(np.float64)
                for target in range(y.shape[1]):
                    reference = Ridge(alpha=1., fit_intercept=True, solver="cholesky").fit(x[fit_rows], y[fit_rows, target])
                    got = x[val_rows] @ saved["coefficients"][fi, target] + saved["intercepts"][fi, target]
                    np.testing.assert_allclose(got, reference.predict(x[val_rows]), rtol=1e-8, atol=1e-9)
                    mse = np.mean((got - y[val_rows, target]) ** 2)
                    np.testing.assert_allclose(mse, saved["fold_mse"][fi, target], rtol=1e-9, atol=1e-10)
                    ridge_checks += 1
    # Independently execute the exact vendored search function over cached scores.
    tree = ast.parse((common.REFERENCE / "sweep.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_ternary_search_layer")
    namespace = {"PrefillResult": object, "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "vendored-ternary-smoke", "exec"), namespace)
    for li, penalty in enumerate(penalties):
        for model in range(models):
            trace = []
            def evaluate(unused, layer, *rest):
                trace.append(layer)
                return -score_cache[layer][li * models + model]
            namespace["_eval_layer"] = evaluate
            layer, value = namespace["_ternary_search_layer"](None, layers, "last", components, None, None, 5)
            assert trace == selected["traces"][str(penalty)][model]
            assert layer == selected["selected"][str(penalty)]["layers"][model]
            np.testing.assert_allclose(-value, selected["selected"][str(penalty)]["validation_mse"][model])
    # A repeated selection must read cached fits instead of fitting Ridge again.
    selection_path.unlink()
    ridge_class = probe.Ridge
    def forbid_refit(*args, **kwargs):
        raise AssertionError("Unexpected Ridge fit during cached selection")
    probe.Ridge = forbid_refit
    try:
        probe.run(0)
    finally:
        probe.Ridge = ridge_class
    cached = json.loads(selection_path.read_text())
    for key in ("selected", "traces", "evaluated_layers", "folds_sha256"):
        assert cached[key] == selected[key]
    duplicate_layers = [layers[0], layers[1], layers[0]]
    x, xe = common.assemble_features(panel, split, duplicate_layers)
    assert x.shape == (n_train, models * components) and xe.shape == (n_test, models * components)
    np.testing.assert_array_equal(x[:, :components], x[:, 2 * components:])
    np.testing.assert_array_equal(xe[:, :components], features[layers[0]][te])

    mod = common.trunk()
    fit_count, reload_count, negative_predictions = 0, 0, 0
    all_predictions, all_validation = {}, {}
    for penalty in penalties:
        selected_layers = cached["selected"][str(penalty)]["layers"]
        x, xe = common.assemble_features(panel, split, selected_layers)
        targets = (data["r"][tr] - penalty * data["c"][tr]).astype(np.float32)
        all_predictions[penalty], all_validation[penalty] = [], []
        for seed in seeds:
            # Real adapted network and optimizer, with toy epoch/batch limits only.
            net = mod.train_mlp(mod.SharedTrunkNet(x.shape[1], models), x, targets,
                                device=args.device, seed=seed, epochs=4, batch_size=16)
            prediction = train.predict(net, xe, args.device)
            order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed)).numpy()
            val = order[int(.85 * len(tr)):]
            vp = train.predict(net, x[val], args.device)
            mse = float(np.mean((vp - targets[val]) ** 2))
            np.testing.assert_allclose(mse, net._best_val_loss, rtol=2e-5, atol=1e-7)
            assert np.isfinite(prediction).all()
            negative_predictions += int((prediction < 0).sum())
            path = train.fitpath(panel, split, penalty, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            common.atomic_npz(path.with_suffix(".npz"), rows=te, prediction=prediction,
                              validation_rows=tr[val], validation_prediction=vp)
            torch.save(net.cpu().state_dict(), path.with_suffix(".pt"))
            common.atomic_json(path, dict(identity=train.fitidentity(panel, split, penalty, seed, selected_layers),
                validation_mse=mse, input_dimensions=x.shape[1],
                predictions_sha256=common.base.sha(path.with_suffix(".npz")),
                checkpoint_sha256=common.base.sha(path.with_suffix(".pt"))))
            assert train.completed(panel, split, penalty, seed, selected_layers) is not None
            restored = mod.SharedTrunkNet(x.shape[1], models).to(args.device)
            restored.load_state_dict(torch.load(path.with_suffix(".pt"), map_location=args.device, weights_only=True))
            restored.eval()
            np.testing.assert_allclose(train.predict(restored, xe, args.device), prediction, rtol=3e-4, atol=3e-5)
            all_predictions[penalty].append(prediction)
            all_validation[penalty].append(mse)
            fit_count += 1
            reload_count += 1
    assert negative_predictions > 0, "Expected unconstrained negative utility predictions, not sigmoid outputs"
    train.assemble(panel, split)
    result = json.loads((destination / "result.json").read_text())
    expected_choices = {}
    with np.load(destination / "predictions.npz") as saved:
        for penalty in penalties:
            keep = sorted(range(10), key=lambda i: (all_validation[penalty][i], seeds[i]))[:5]
            assert result["ensembles"][str(penalty)]["seeds"] == [seeds[i] for i in keep]
            predicted = np.mean(np.asarray([all_predictions[penalty][i] for i in keep]), axis=0)
            chosen = predicted.argmax(axis=1)
            np.testing.assert_array_equal(saved[f"prediction_{penalty:g}"], predicted)
            np.testing.assert_array_equal(saved[f"choice_{penalty:g}"], chosen)
            utility = float(np.mean(data["r"][te, chosen] - penalty * data["c"][te, chosen]))
            np.testing.assert_allclose(result["scores"][str(penalty)]["utility"], utility, rtol=0, atol=1e-12)
            expected_choices[penalty] = chosen.copy()
    # Changing held-out realized costs must change evaluation, never model choice.
    # This sentinel detects subtracting test costs a second time from predicted utility.
    old_costs = data["c"][te].copy()
    data["c"][te, expected_choices[2]] += 1000.
    try:
        train.assemble(panel, split)
        changed_result = json.loads((destination / "result.json").read_text())
        with np.load(destination / "predictions.npz") as saved:
            for penalty in penalties:
                np.testing.assert_array_equal(saved[f"choice_{penalty:g}"], expected_choices[penalty])
                expected = np.mean(data["r"][te, expected_choices[penalty]]
                                   - penalty * data["c"][te, expected_choices[penalty]])
                np.testing.assert_allclose(changed_result["scores"][str(penalty)]["utility"], expected, rtol=0, atol=1e-10)
    finally:
        data["c"][te] = old_costs
    train.assemble(panel, split)
    report = dict(passed=True, synthetic_only=True, device=args.device,
        scope="Actual utility probe, reference search, fitting core, prediction, checkpoint reload and ensemble assembly; no paper experiments.",
        train_rows=n_train, test_rows=n_test, targets=models, feature_block_width=components,
        layers=layers, penalties=penalties, seeds=seeds, toy_epochs=4, toy_batch_size=16,
        independent_ridge_predictions=ridge_checks, reference_ternary_paths=models * len(penalties),
        cached_selection_identical=True, repeated_blocks_preserved=True,
        trained_networks=fit_count, checkpoint_reloads=reload_count,
        negative_raw_predictions=negative_predictions, top_five_seed_retention=True,
        cost_applied_once=True, heldout_cost_changes_do_not_change_routes=True,
        exclusions=["Full-data integrity/preflight", "Encoder extraction", "PCA fitting", "GPU worker scheduling", "Full experimental reproduction"])
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Report: " + str((work / "report.json").relative_to(ROOT)))


if __name__ == "__main__":
    main()
