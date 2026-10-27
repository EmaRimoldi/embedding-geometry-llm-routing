#!/usr/bin/env python3
"""Compare newly computed main-baseline records with the frozen study evidence.

This audits supplied result files, not their execution history or omitted predictions.
Exit 0: complete and within tolerance; 1: differences/error; 2: incomplete coverage.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTOL, ATOL = 1e-10, 1e-12
OPERATIONAL_FIELDS = {"seconds", "slurm_job_id"}
PROVENANCE_FIELDS = {"code_sha256", "config_sha256", "trainer_sha256", "protocol_sha256"}
EXACT_SELECTION_FIELDS = {"alpha", "alphas", "candidate_alphas", "k", "K", "tau", "temperature",
                          "architecture", "layers", "seeds", "n_clusters", "early_stopping"}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def expected_paths(tasks):
    paths = {}
    for task in tasks:
        dataset = {"routerbench-0shot": "routerbench", "sprout-sixfam": "sprout"}[task["panel"]]
        encoder = {"gte": "gte-qwen2-7b", "reader": "qwen2.5-7b-instruct/layer_21"}[task["representation"]]
        kinds = ["knn", "prior", "clusters", "prox", "ridge", "mlp_reduced"]
        if task["split"].startswith("iid_"):
            kinds.append("lookups")
        for kind in kinds:
            path = "/".join(["baselines", encoder, dataset, task["split"], kind, "result.json"])
            if path in paths:
                raise ValueError("Duplicate baseline task: " + path)
            paths[path] = kind
    return paths


def json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def compare_tree(actual, expected, path, counts, differences):
    def difference(kind="value", **extra):
        differences.append(dict(path=path, kind=kind, fresh=json_value(actual),
                                reference=json_value(expected), within_tolerance=False, **extra))
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            difference("type")
            return
        if set(actual) != set(expected):
            differences.append(dict(path=path, kind="dictionary_keys",
                                    missing=sorted(set(expected)-set(actual)),
                                    extra=sorted(set(actual)-set(expected)), within_tolerance=False))
        for key in sorted(set(actual) & set(expected)):
            compare_tree(actual[key], expected[key], path + [key], counts, differences)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            difference("list_shape")
            return
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_tree(left, right, path + [index], counts, differences)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        counts["other_values"] += 1
        if type(actual) is not type(expected) or actual != expected:
            difference()
    elif isinstance(expected, (int, float)):
        counts["numeric_values"] += 1
        if type(actual) is not type(expected) or not math.isfinite(actual) or not math.isfinite(expected):
            difference("numeric_type_or_finiteness")
        elif actual == expected:
            counts["exact_numeric_values"] += 1
        else:
            discrete = path[:1] == ["selections"] and any(key in EXACT_SELECTION_FIELDS for key in path)
            close = (not isinstance(expected, int) and not discrete
                     and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL))
            differences.append(dict(path=path, kind="numeric", fresh=actual, reference=expected,
                                    absolute_error=abs(actual-expected), within_tolerance=close))
    else:
        raise TypeError("Unsupported reference value")


def audit(fresh, reference, expected, transitions):
    """Compare decoded records; transitions must already be verified against source bytes."""
    if not expected:
        raise ValueError("The expected baseline task set must not be empty")
    counts = Counter(numeric_values=0, exact_numeric_values=0, other_values=0)
    differences, records = [], []
    coverage, methods = Counter(), Counter()
    for path in sorted(set(fresh) & set(expected)):
        if path not in reference:
            raise ValueError("Missing reference: " + path)
        actual, original = fresh[path], reference[path]
        start = len(differences)
        compare_tree({k: v for k, v in actual.items() if k not in OPERATIONAL_FIELDS},
                     {k: v for k, v in original.items() if k not in OPERATIONAL_FIELDS},
                     [], counts, differences)
        for item in differences[start:]:
            field = item["path"]
            pair = (item.get("reference"), item.get("fresh"))
            known = (len(field) == 1 and field[0] in PROVENANCE_FIELDS
                     and all(isinstance(value, str) for value in pair) and pair in transitions)
            item["record"] = path
            item["classification"] = "recorded_release_provenance" if known else "scientific_or_unexplained"
            if known:
                item["source_path"] = transitions[pair]
        values = actual.get("results", {})
        method_counts = Counter(name for policies in values.values() if isinstance(policies, dict)
                                for name in policies) if isinstance(values, dict) else Counter()
        methods.update(method_counts)
        coverage[expected[path]] += 1
        records.append(dict(path=path, kind=expected[path], method_scores=sum(method_counts.values())))
    scientific = [item for item in differences if item["classification"] != "recorded_release_provenance"]
    violations = [item for item in scientific if not item["within_tolerance"]]
    missing = sorted(set(expected)-set(fresh))
    complete = not missing
    passed = complete and not violations
    return dict(status="differences" if violations else "matched" if complete else "incomplete",
                passed=passed, coverage_complete=complete, expected_records=len(expected),
                compared_completed_records=len(records), completed_records_by_kind=dict(coverage),
                expected_records_by_kind=dict(Counter(expected.values())), method_scores=sum(methods.values()),
                method_scores_by_name=dict(methods), counts=dict(counts),
                relative_tolerance=RTOL, absolute_tolerance=ATOL,
                excluded_operational_fields=sorted(OPERATIONAL_FIELDS),
                exact_scientific_match=bool(records) and not scientific,
                scientific_fields_within_tolerance=bool(records) and not violations,
                compared_record_fields_identical=bool(records) and not differences,
                recorded_provenance_differences=len(differences)-len(scientific),
                scientific_or_unexplained_differences=len(scientific), beyond_tolerance=len(violations),
                records=records, differences=differences, missing_records=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-results", required=True, type=Path,
                        help="Results directory of the separate new run, containing baselines/.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/fresh-baselines.json")
    args = parser.parse_args()
    frozen = (ROOT / "results/evidence").resolve()
    fresh_results = args.fresh_results.resolve()
    output = args.output.resolve()
    if fresh_results == frozen or frozen in fresh_results.parents:
        parser.error("Frozen evidence is not a fresh run.")
    if output == frozen or frozen in output.parents or output in {ROOT / "release_provenance/release_manifest.json"}:
        parser.error("Output must not overwrite frozen evidence or the release manifest.")
    baseline_inputs = fresh_results / "baselines"
    if output == baseline_inputs or baseline_inputs in output.parents or output == Path(__file__).resolve():
        parser.error("Output must not overwrite fresh baseline inputs or the auditor.")
    release = json.loads((ROOT / "release_provenance/release_manifest.json").read_text())["files"]
    for source_root in {ROOT, fresh_results.parent}:
        try:
            relative_output = str(output.relative_to(source_root))
        except ValueError:
            continue
        if relative_output in release:
            parser.error("Output must not overwrite a pinned project file.")
    def pinned(relative):
        content = (ROOT / relative).read_bytes()
        if digest(content) != release[relative]["sha256"]:
            raise ValueError("Reference configuration checksum differs: " + relative)
        return content
    tasks = json.loads(pinned("configs/baseline_tasks.json"))
    expected = expected_paths(tasks)
    if len(tasks) != 46 or len(expected) != 296:
        raise ValueError("The prescribed main-baseline task grid differs")
    changes = json.loads(pinned("release_provenance/source_changes.json"))
    evidence_bytes = pinned("results/release_evidence_manifest.json")
    evidence = {item["exported_relative_path"]: item["exported_sha256"]
                for item in json.loads(evidence_bytes)["records"]}
    transitions = {}
    for item in changes:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source transition path")
        source = fresh_results.parent / relative
        if source.is_file() and digest(source.read_bytes()) == item["release_sha256"]:
            transitions[(item["original_sha256"], item["release_sha256"])] = item["path"]
    fresh, reference, fingerprints = {}, {}, {}
    for path in expected:
        original = ROOT / "results/evidence" / path
        if not original.is_file() or "results/evidence/" + path not in evidence:
            raise ValueError("Missing pinned baseline reference: " + path)
        actual = fresh_results / path
        if not actual.is_file():
            continue
        before, after = original.read_bytes(), actual.read_bytes()
        if digest(before) != evidence["results/evidence/" + path]:
            raise ValueError("Frozen result checksum differs: " + path)
        reference[path], fresh[path] = json.loads(before), json.loads(after)
        fingerprints[path] = dict(fresh_file_sha256=digest(after), reference_file_sha256=digest(before))
    report = audit(fresh, reference, expected, transitions)
    for record in report["records"]:
        record.update(fingerprints[record["path"]])
    from datetime import datetime, timezone
    report.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  auditor_sha256=digest(Path(__file__).read_bytes()),
                  reference_evidence_manifest_sha256=digest(evidence_bytes),
                  transition_registry="release_provenance/source_changes.json",
                  scope="Supplied main-baseline scores, selections, cohorts and split identities compared with pinned original records. This command does not attest execution history, compare omitted per-query predictions or test paper printed precision.")
    context = fresh_results.parent / ".rerun/context.json"
    if context.is_file():
        run = json.loads(context.read_text())
        report["reported_run_context"] = {key: run[key] for key in ["release", "commit", "source_zip_sha256"] if key in run}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"records", "differences", "missing_records"}}, indent=2))
    return 0 if report["passed"] else 1 if report["beyond_tolerance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
