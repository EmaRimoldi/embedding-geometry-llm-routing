#!/usr/bin/env python3
"""Verify anonymous evidence and rebuild aggregates using only Python's standard library.

This re-aggregates saved fold-level scores. It does not rerun models, reconstruct
query-level decisions, or recompute bootstrap intervals from excluded arrays.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/evidence"
METRICS = ("utility", "reward", "cost")


def method_metadata(method):
    """Label the information regime without removing diagnostic or oracle rows."""
    if method == "Realized oracle":
        return dict(method_role="realized_oracle",
                    required_information="Recorded rewards and costs of every candidate on each test query; cost penalty")
    if method == "Family lookup":
        return dict(method_role="known_metadata_diagnostic",
                    required_information="Known test family; source family labels and recorded source rewards/costs; cost penalty")
    if method == "Family-topic lookup":
        return dict(method_role="known_metadata_diagnostic",
                    required_information="Known test native cell and family; source cell/family labels and recorded source rewards/costs; cost penalty")
    information = "Test prompt embedding; recorded source rewards and costs; cost penalty"
    if method == "Utility-cluster control":
        return dict(method_role="diagnostic_clustering_control", required_information=information)
    if method == "Best single model":
        information = "Recorded source rewards and costs; cost penalty; no test features or outcomes"
    elif method == "Family Adaptation":
        information = "Test prompt embedding; source family labels and recorded source rewards/costs; cost penalty"
    elif method == "SharedTrunkNet per-model utility":
        information = "Test prompt embeddings at selected Qwen layers; recorded source rewards/costs; cost penalty"
    return dict(method_role="source_trained_router", required_information=information)


def load_json(path):
    return json.loads(path.read_text())


def read_csv(relative):
    with (EVIDENCE / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number_key(value):
    try:
        return format(float(value), ".15g")
    except (ValueError, TypeError):
        return str(value)


def row_key(row, keys):
    return tuple(number_key(row[key]) if key in {"lambda_value", "k", "layer"}
                 else str(row[key]) for key in keys)


def check_close(actual, expected, context):
    actual, expected = float(actual), float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"Non-finite value: {context}")
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Mismatch {context}: {actual} != {expected}")


def compare(rows, reference, keys, metrics):
    index = {row_key(row, keys): row for row in reference}
    if len(index) != len(reference):
        raise ValueError("Duplicate reference rows")
    actual = {row_key(row, keys): row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(index):
        raise ValueError(f"Aggregate row keys differ: {set(actual) ^ set(index)}")
    for key, row in actual.items():
        for metric in metrics:
            check_close(row[metric], index[key][metric], (key, metric))
        for field in ("folds", "folds_completed", "folds_expected", "complete"):
            if field in row and field in index[key] and str(row[field]) != str(index[key][field]):
                raise ValueError(f"Fold completeness differs: {key} {field}")


def pool(rows, keys, weight="n_test", metrics=METRICS, geometry=False):
    groups = defaultdict(list)
    for row in rows:
        groups[row_key(row, keys)].append(row)
    aggregates = []
    for key, group in sorted(groups.items()):
        total = sum(int(row[weight]) for row in group)
        if total <= 0:
            raise ValueError(f"No test observations: {key}")
        item = dict(zip(keys, key))
        item.update({metric: sum(float(row[metric]) * int(row[weight])
                                 for row in group) / total for metric in metrics})
        if geometry:
            item["folds"] = len(group)
        else:
            panel = item.get("dataset", item.get("panel"))
            expected = 5 if item["protocol"] == "iid" else (7 if panel == "routerbench-0shot" else 6)
            item.update(folds_completed=len(group), folds_expected=expected,
                        complete=len(group) == expected)
        aggregates.append(item)
    return aggregates


def utility_identity(rows):
    for row in rows:
        check_close(float(row["reward"]) - float(row["lambda_value"]) * float(row["cost"]),
                    row["utility"], "reward - penalty * cost")


def verify_evidence():
    manifest = load_json(ROOT / "results/release_evidence_manifest.json")
    for record in manifest["records"]:
        path = ROOT / record["exported_relative_path"]
        if not path.is_file():
            raise ValueError(f"Missing evidence: {path.relative_to(ROOT)}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["exported_sha256"]:
            raise ValueError(f"Evidence hash mismatch: {path.relative_to(ROOT)}")
    return {"files": len(manifest["records"]), "bytes": sum(r["bytes"] for r in manifest["records"])}


def final_utility_rows():
    rows = []
    base = EVIDENCE / "nvidia_layer_validation/utility_v1"
    for dataset in ("routerbench", "sprout"):
        for path in sorted((base / dataset).glob("*/result.json")):
            record = load_json(path)
            identity = record["identity"]
            for penalty, values in record["scores"].items():
                rows.append(dict(panel=identity["panel"], split=identity["split"],
                                 protocol=identity["split"].split("_")[0],
                                 lambda_value=penalty, n_test=record["n_test"],
                                 **{key: values[key] for key in (*METRICS, "equal_family")}))
    reference = read_csv("nvidia_layer_validation/utility_v1/summary/scores.csv")
    compare(rows, reference, ["panel", "split", "lambda_value"], [*METRICS, "equal_family", "n_test"])
    if len(rows) != 299:
        raise ValueError(f"Expected all 299 final utility fold/penalty scores, got {len(rows)}")
    return rows


def emit_plots(path, rows):
    """Portable SVG overview, with every penalty and negative utility retained."""
    colors = ["#2166ac", "#b2182b", "#00876c", "#88419d", "#d95f02", "#555555", "#3e7e00", "#a6761d", "#c51b7d", "#00a6b5"]
    outputs = []
    for panel in ("routerbench-0shot", "sprout-sixfam"):
        series = defaultdict(list)
        for row in rows:
            if row["dataset"] == panel and row["protocol"] == "ood":
                # A readable overview; the CSV retains every method and all penalties.
                if row["method"] in {"kNN-100", "Family Adaptation", "MLP", "Ridge", "SharedTrunkNet per-model utility"}:
                    series[row["encoder"] + " / " + row["method"]].append(row)
        values = [float(row["utility"]) for group in series.values() for row in group]
        lo, hi = min(values), max(values)
        padding = (hi - lo) * .06 or .05
        lo -= padding
        hi += padding
        width, height, left, top, plot_width, plot_height = 1220, 520, 80, 48, 580, 400
        xmax = max(float(row["lambda_value"]) for group in series.values() for row in group)
        def x(v): return left + math.log1p(v) / math.log1p(xmax) * plot_width
        def y(v): return top + plot_height * (hi - v) / (hi - lo)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="white"/>',
               '<g font-family="sans-serif" font-size="12" fill="#222">',
               f'<text x="80" y="25" font-size="17">{html.escape(panel)}: held-out-family utility</text>']
        for i in range(6):
            val = lo + (hi - lo) * i / 5
            svg.append(f'<path d="M {left} {y(val):.3f} h {plot_width}" stroke="#ddd"/>')
            svg.append(f'<text x="70" y="{y(val)+4:.3f}" text-anchor="end">{val:.3f}</text>')
        for penalty in [0, 1, 5, 20, 100, 500]:
            svg.append(f'<text x="{x(penalty):.3f}" y="471" text-anchor="middle">{penalty}</text>')
        svg.append('<text x="370" y="495" text-anchor="middle">Cost penalty (position proportional to log(1 + penalty))</text>')
        svg.append('<text x="20" y="250" transform="rotate(-90 20 250)" text-anchor="middle">Query-weighted mean utility</text>')
        for index, (label, group) in enumerate(sorted(series.items())):
            color = colors[index % len(colors)]
            ordered = sorted(group, key=lambda row: float(row["lambda_value"]))
            points = " ".join(f'{x(float(row["lambda_value"])):.3f},{y(float(row["utility"])):.3f}' for row in ordered)
            svg.append(f'<polyline points="{points}" stroke="{color}" stroke-width="1.7" fill="none"/>')
            for row in ordered:
                svg.append(f'<circle cx="{x(float(row["lambda_value"])):.3f}" cy="{y(float(row["utility"])):.3f}" r="2.7" fill="{color}"/>')
            svg.append(f'<path d="M 690 {60+index*30} h 25" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<text x="725" y="{64+index*30}">{html.escape(label)}</text>')
        svg.extend(['<text x="690" y="440" font-size="11">Rebuilt from exported fold-level scores.</text>',
                    '<text x="690" y="458" font-size="11">Overview figure; no intervals recomputed.</text>', '</g></svg>'])
        target = path / "figures" / f"{panel}_ood_utility.svg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(svg) + "\n")
        outputs.append(str(target.relative_to(path)))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    parser.add_argument("--verify-only", action="store_true", help="Only verify exported evidence hashes; write nothing")
    parser.add_argument("--plots", action="store_true", help="Also write two SVG utility overview charts")
    args = parser.parse_args()
    verified = verify_evidence()
    if args.verify_only:
        print(json.dumps({"evidence_integrity": "passed", **verified}, indent=2))
        return
    output = args.output.resolve()
    if output == EVIDENCE.resolve() or EVIDENCE.resolve() in output.parents:
        raise ValueError("Output must not overwrite frozen evidence")
    reports, aggregates = [], {}
    specs = [
        ("baselines_original", "baselines/_summary/per_split.csv", "baselines/_summary/scores.csv", ["dataset", "protocol", "encoder", "method", "lambda_value"], "n_test"),
        ("baselines_reduced", "baselines/_summary_reduced/per_split.csv", "baselines/_summary_reduced/scores.csv", ["dataset", "protocol", "encoder", "method", "lambda_value"], "n_test"),
        ("ridge", "baselines/_summary/ridge_per_split.csv", "baselines/_summary/ridge_scores.csv", ["dataset", "protocol", "encoder", "method", "lambda_value"], "n_test"),
        ("layer_sweep", "layer_sweep/summary/per_split.csv", "layer_sweep/summary/scores.csv", ["dataset", "protocol", "layer", "method", "lambda_value"], "n"),
        ("nvidia_earlier_controls", "nvidia/summary/per_split.csv", "nvidia/summary/scores.csv", ["dataset", "protocol", "method", "lambda_value"], "n_test"),
    ]
    for name, source, reference, keys, weight in specs:
        rows = read_csv(source)
        utility_identity(rows)
        pooled = pool(rows, keys, weight)
        compare(pooled, read_csv(reference), keys, METRICS)
        write_csv(output / (name + ".csv"), pooled)
        aggregates[name] = pooled
        reports.append(dict(name=name, source=source, fold_scores=len(rows), aggregate_rows=len(pooled), verified_against=reference))
    geo_keys = ["encoder", "dataset", "subset", "mode", "k"]
    geo_metrics = ["same_family_share", "random_same_family_share", "largest_family_share", "same_format_share", "relative_reward_mse", "cost_mse"]
    geo_rows = read_csv("geometry/_summary/neighborhoods.csv")
    geo = pool(geo_rows, geo_keys, metrics=geo_metrics, geometry=True)
    compare(geo, read_csv("geometry/_summary/pooled_query_weighted.csv"), geo_keys, geo_metrics)
    write_csv(output / "geometry_pooled.csv", geo)
    reports.append(dict(name="geometry_pooled", fold_scores=len(geo_rows), aggregate_rows=len(geo)))
    final_rows = final_utility_rows()
    utility_identity(final_rows)
    final_keys = ["panel", "protocol", "lambda_value"]
    final = pool(final_rows, final_keys)
    reference = load_json(EVIDENCE / "nvidia_layer_validation/utility_v1/summary/report.json")
    for summary in reference["summary"]:
        group = [r for r in final if r["panel"] == summary["panel"] and r["protocol"] == summary["protocol"]]
        if len(group) != 13 or not all(r["complete"] for r in group):
            raise ValueError("Final utility comparison must contain all folds and all 13 penalties")
        compare(group, [dict(panel=summary["panel"], protocol=summary["protocol"], **r) for r in summary["per_penalty"]], final_keys, ["utility"])
        check_close(sum(r["utility"] for r in group)/13, summary["mean_utility"], "mean across final utility penalties")
    write_csv(output / "nvidia_final_per_split.csv", final_rows)
    write_csv(output / "nvidia_final_per_penalty.csv", final)
    reports.append(dict(name="nvidia_final_per_model_utility", source="23 exported result.json files", fold_scores=len(final_rows), aggregate_rows=len(final), verified_against="utility_v1 saved fold CSV and aggregate report"))
    # Main comparison uses completed reduced-grid MLP and final per-target SharedTrunkNet.
    columns = ["dataset", "protocol", "encoder", "method", "lambda_value", "utility", "reward", "cost"]
    combined = [{key: row[key] for key in columns} for name in ["baselines_reduced", "ridge"]
                for row in aggregates[name] if row["complete"]]
    combined += [dict(dataset=row["panel"], protocol=row["protocol"], encoder="Qwen selected layers",
                      method="SharedTrunkNet per-model utility", lambda_value=row["lambda_value"],
                      **{key: row[key] for key in METRICS}) for row in final]
    for row in combined:
        row.update(method_metadata(row["method"]))
    write_csv(output / "main_comparison_all_penalties.csv", combined)
    by_method = defaultdict(list)
    for row in combined:
        by_method[row["dataset"], row["protocol"], row["encoder"], row["method"]].append(row)
    means = []
    for key, rows in sorted(by_method.items()):
        if len(rows) != 13:
            raise ValueError(f"Incomplete penalty grid in main comparison: {key}")
        means.append(dict(zip(["dataset", "protocol", "encoder", "method"], key),
                          penalties=13, mean_utility=sum(float(row["utility"]) for row in rows)/13,
                          **method_metadata(key[3])))
    write_csv(output / "main_comparison_mean_utility.csv", means)
    figures = emit_plots(output, combined) if args.plots else []
    report = dict(status="passed", evidence_verified=verified, reconstructed=reports,
                  numerical_tolerance=dict(relative=1e-10, absolute=1e-12), figures=figures,
                  interpretation="Saved fold-level score aggregation; final utility additionally reconstructed from exported result JSONs. No model execution, per-query decision recomputation, or bootstrap resampling.",
                  summary_only="Confidence intervals, format controls, family-transfer tests and other saved audit reports are preserved evidence. Their underlying arrays are excluded from this compact package.",
                  canonical_nvidia="results/evidence/nvidia_layer_validation/utility_v1; results/evidence/nvidia contains earlier controls. Earlier inference timings do not describe the final per-target pipeline.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
