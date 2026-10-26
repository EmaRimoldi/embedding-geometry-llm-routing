import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rb import oracle_prompt as O

MODELS = ["cheap", "mid", "strong"]


def test_cell_table_mean_utility():
    r = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], float); c = np.array([[1e-3, 2e-3, 4e-3]] * 3)
    t = O.cell_table(r, c, np.array(["a", "a", "b"]), MODELS, lam=100.0)
    assert set(t) == {"a", "b"} and np.isclose(t["a"]["strong"], 1 - 0.4) and np.isclose(t["b"]["cheap"], -0.1)


def test_neighbour_rows_orders_by_cosine():
    xt = np.eye(3, dtype=np.float32); xq = np.array([0.1, 0.9, 0.0], np.float32); xq /= np.linalg.norm(xq)
    nb = O.neighbour_rows(xq, xt, k=2)
    assert nb.tolist() == [1, 0]


def test_format_prompt_contains_rows_and_truncates():
    nbs = [{"text": "x" * 1000, "rewards": [1, 0, 1], "costs": [1e-3, 2e-3, 4e-3], "cell": "a"}]
    p = O.format_prompt("What is 2+2?", nbs, "a", {"cheap": 0.9, "mid": 0.8, "strong": 0.7}, MODELS, {"cheap": 0.1, "mid": 0.5, "strong": 2.0}, lam=100.0)
    assert "What is 2+2?" in p and "x" * 300 in p and "x" * 301 not in p and "decision_source" in p and "strong" in p


def test_parse_decision_accepts_fenced_json_and_names():
    txt = 'Thinking...\n```json\n{"decision_source": "reasoning", "chosen_model": "strong", "lookup_choice": "cheap", "rationale": "hard"}\n```'
    d = O.parse_decision(txt, MODELS)
    assert d == {"decision_source": "reasoning", "chosen_model": 2, "lookup_choice": 0, "rationale": "hard"}


def test_parse_decision_rejects_garbage():
    import pytest
    with pytest.raises(ValueError):
        O.parse_decision("no json here", MODELS)
    with pytest.raises(ValueError):
        O.parse_decision('{"decision_source": "magic", "chosen_model": "strong"}', MODELS)


def test_parse_decision_tolerates_braces_in_rationale():
    txt = '{"decision_source": "reasoning", "chosen_model": "strong", "lookup_choice": "cheap", "rationale": "the set {1,2} is small"}'
    d = O.parse_decision(txt, MODELS)
    assert d["chosen_model"] == 2


def test_machine_counterfactuals():
    r = np.array([[1, 0, 1], [1, 0, 1], [0, 0, 1]], float); c = np.array([[1e-3, 2e-3, 4e-3]] * 3)
    cf = O.machine_counterfactuals(r, c, np.array([0, 1]), {"cheap": 0.5, "mid": 0.1, "strong": 0.6}, MODELS, lam=0.0)
    assert cf["knn_choice"] == 0 and cf["cell_choice"] == 2      # kNN: cheap ties strong on reward, cheaper wins


def test_record_schema_roundtrip_and_meta():
    args = ("q1", 100.0, "reasoning", "abc", {"decision_source": "lookup", "chosen_model": 1, "lookup_choice": 1, "rationale": ""},
            {"knn_choice": 1, "cell_choice": 2}, {"reward": [1, 0, 1], "cost": [1e-3, 2e-3, 4e-3]}, "raw", {"input_tokens": 10, "output_tokens": 5})
    rec = O.record(*args)
    assert json.loads(json.dumps(rec))["chosen_model"] == 1 and rec["query_id"] == "q1" and "utility_chosen" in rec
    assert "model" not in rec                                          # meta is optional
    rec = O.record(*args, meta={"model": "Qwen/Qwen2.5-32B-Instruct", "backend": "openai_compatible"})
    assert rec["model"] == "Qwen/Qwen2.5-32B-Instruct" and rec["backend"] == "openai_compatible"
    assert rec["chosen_model"] == 1 and json.loads(json.dumps(rec))["backend"] == "openai_compatible"


def test_parse_decision_rejects_boolean_model_index():
    """bool subclasses int: {"chosen_model": true} must not silently mean model 1."""
    import pytest
    with pytest.raises(ValueError):
        O.parse_decision('{"decision_source": "lookup", "chosen_model": true}', MODELS)
    assert O.parse_decision('{"decision_source": "lookup", "chosen_model": 1}', MODELS)["chosen_model"] == 1


def test_machine_counterfactuals_respects_availability():
    r = np.array([[1, 0, 1], [1, 0, 1], [0, 0, 1]], float); c = np.array([[1e-3, 2e-3, 4e-3]] * 3)
    cu = {"cheap": 0.5, "mid": 0.1, "strong": 0.6}
    cf = O.machine_counterfactuals(r, c, np.array([0, 1]), cu, MODELS, lam=0.0, available=[False, True, True])
    assert cf["knn_choice"] == 2 and cf["cell_choice"] == 2        # cheap was never run on this query
