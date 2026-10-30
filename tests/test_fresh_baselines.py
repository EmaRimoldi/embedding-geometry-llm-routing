import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fresh_baselines", ROOT / "scripts/compare_baseline_results.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)
RECORD = "baselines/gte-qwen2-7b/routerbench/iid_0/ridge/result.json"


class FreshBaselineChecks(unittest.TestCase):
    def setUp(self):
        self.reference = {RECORD: dict(
            panel="routerbench-0shot", split="iid_0", representation="gte",
            results={"0": {"ridge": dict(utility=.5, reward=.6, cost=.1)}},
            selections={"0": dict(alpha=.01, candidate_alphas=[.001, .01],
                                   validation_utilities=[.2, .5], validation_queries=10)},
            n_train=20, n_test=10, code_sha256="a" * 64, seconds=1.)}
        self.fresh = copy.deepcopy(self.reference)
        self.expected = {RECORD: "ridge"}

    def report(self, transitions=None):
        return CHECKS.audit(self.fresh, self.reference, self.expected, transitions or {})

    def test_matching_records_and_elapsed_time(self):
        self.fresh[RECORD]["seconds"] = 900.
        report = self.report()
        self.assertTrue(report["passed"])
        self.assertTrue(report["exact_scientific_match"])
        self.assertEqual(report["counts"]["numeric_values"], report["counts"]["exact_numeric_values"])

    def test_missing_record_never_passes(self):
        self.fresh.clear()
        report = self.report()
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["passed"])
        self.assertFalse(report["exact_scientific_match"])
        self.assertFalse(report["scientific_fields_within_tolerance"])
        self.assertEqual(report["missing_records"], [RECORD])

    def test_empty_expected_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            CHECKS.audit({}, {}, {}, {})

    def test_changed_score_is_not_hidden_by_identical_selections(self):
        self.fresh[RECORD]["results"]["0"]["ridge"]["utility"] = .51
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertEqual(report["beyond_tolerance"], 1)

    def test_selected_alpha_requires_exact_agreement(self):
        self.fresh[RECORD]["selections"]["0"]["alpha"] += 1e-13
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertEqual(report["beyond_tolerance"], 1)

    def test_selected_prox_inverse_temperature_requires_exact_agreement(self):
        path = RECORD.replace("/ridge/", "/prox/")
        reference = dict(
            selections={"0": dict(inverse_temperature=20., validation_utilities=[.2, .5])},
            results={"0": {"km_prox32_validated": dict(utility=.5)}})
        actual = copy.deepcopy(reference)
        actual["selections"]["0"]["inverse_temperature"] += 1e-10
        report = CHECKS.audit({path: actual}, {path: reference}, {path: "prox"}, {})
        self.assertFalse(report["passed"])
        self.assertEqual(report["beyond_tolerance"], 1)
        self.assertEqual(report["differences"][0]["path"],
                         ["selections", "0", "inverse_temperature"])

    def test_clustering_candidate_grid_requires_exact_agreement(self):
        path = RECORD.replace("/ridge/", "/clusters/")
        reference = dict(candidates=[[32, 1, .01]],
                         selections={"0": dict(alpha=.01, validation_utility=.5)})
        actual = copy.deepcopy(reference)
        actual["candidates"][0][2] += 1e-13
        report = CHECKS.audit({path: actual}, {path: reference}, {path: "clusters"}, {})
        self.assertFalse(report["passed"])
        self.assertEqual(report["beyond_tolerance"], 1)
        self.assertEqual(report["differences"][0]["path"], ["candidates", 0, 2])

    def test_small_score_difference_is_retained(self):
        self.fresh[RECORD]["results"]["0"]["ridge"]["utility"] += 1e-13
        report = self.report()
        self.assertTrue(report["passed"])
        self.assertFalse(report["exact_scientific_match"])
        self.assertEqual(len(report["differences"]), 1)
        self.assertGreater(report["differences"][0]["absolute_error"], 0.)

    def test_invalid_values_shapes_and_keys(self):
        for case in ["boolean_count", "nan", "infinity", "missing_key", "shape"]:
            with self.subTest(case=case):
                self.fresh = copy.deepcopy(self.reference)
                record = self.fresh[RECORD]
                if case == "boolean_count":
                    record["n_test"] = True
                elif case in {"nan", "infinity"}:
                    record["results"]["0"]["ridge"]["utility"] = float("nan" if case == "nan" else "inf")
                elif case == "missing_key":
                    del record["results"]
                else:
                    record["selections"]["0"]["candidate_alphas"].pop()
                report = self.report()
                self.assertFalse(report["passed"])
                # Even nonfinite differences can be saved as valid JSON.
                import json
                json.dumps(report, allow_nan=False)

    def test_registered_provenance_is_preserved_as_a_difference(self):
        self.fresh[RECORD]["code_sha256"] = "b" * 64
        report = self.report({("a" * 64, "b" * 64): "src/baselines/run_ridge.py"})
        self.assertTrue(report["passed"])
        self.assertTrue(report["exact_scientific_match"])
        self.assertFalse(report["compared_record_fields_identical"])
        self.assertEqual(report["recorded_provenance_differences"], 1)
        self.assertEqual(report["differences"][0]["fresh"], "b" * 64)

    def test_unregistered_source_hash_is_rejected(self):
        self.fresh[RECORD]["code_sha256"] = "b" * 64
        self.assertFalse(self.report()["passed"])

    def test_provenance_registry_does_not_excuse_a_nested_result_change(self):
        self.reference[RECORD]["results"]["0"]["code_sha256"] = "a" * 64
        self.fresh = copy.deepcopy(self.reference)
        self.fresh[RECORD]["results"]["0"]["code_sha256"] = "b" * 64
        self.assertFalse(self.report({("a" * 64, "b" * 64): "source.py"})["passed"])

    def test_duplicate_tasks_cannot_reduce_expected_coverage(self):
        task = dict(panel="routerbench-0shot", split="iid_0", representation="gte")
        self.assertEqual(len(CHECKS.expected_paths([task])), 7)
        with self.assertRaises(ValueError):
            CHECKS.expected_paths([task, task])

    def test_cli_rejects_frozen_evidence_and_overwriting_fresh_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "results"
            cases = [["--fresh-results", str(ROOT / "reproducibility/evidence")],
                     ["--fresh-results", str(fresh), "--output", str(fresh / RECORD)],
                     ["--fresh-results", str(fresh), "--output", str(ROOT / "configs/baseline_tasks.json")]]
            for arguments in cases:
                with self.subTest(arguments=arguments), patch("sys.argv", ["auditor", *arguments]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        CHECKS.main()
                    self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
