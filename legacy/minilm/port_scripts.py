"""minilm/ study: make the representation-bound scripts read the representation from the panel
context instead of hard-coding GTE, so that the same scripts run on MiniLM (E4_REP) without
touching the GTE outputs (E4_RESULTS_ROOT / E4_OUT).  Exact-string replacements with asserts;
with the environment variables unset every script behaves exactly as before (REP_NAME =
"gte-qwen2-7b", OUT_OVERRIDE = None).  Originals are copied to minilm/scripts_backup/.
Idempotent: a file already carrying the new import is skipped."""
from __future__ import annotations
import shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; BK = ROOT / "minilm" / "scripts_backup"
FR = 'find_rep("routerbench-0shot", "computed/gte-qwen2-7b@prompt")'
IMP_FULL = "from rb.panelctx import OUT_OVERRIDE, PANEL, REP, REP_NAME"
EDITS = {
 "src/rb/data.py": [
  ('    "llmrouterbench-performance": (RUNTIME_ROOT / "artifacts/performance-paper.npz",',
   '    # minilm/ study (PI, 2026-09-16): the same release restricted to the rows with <= 512 tokens under the\n'
   '    # MiniLM (BERT) tokenizer; keep mask in features/llmrouterbench-performance-cost-le512/keep_mask.npz.\n'
   '    "llmrouterbench-performance-cost-le512": (RUNTIME_ROOT / "artifacts/performance-cost.npz",\n'
   '                                              "LLMRouterBench performance-cost, rows <= 512 BERT tokens (SWE-bench drops out): 10,911 queries, 13 models, 9 datasets"),\n'
   '    "llmrouterbench-performance": (RUNTIME_ROOT / "artifacts/performance-paper.npz",')],
 "src/rb/panelctx.py": [
  ("import re\n", "import re\nfrom pathlib import Path\n"),
  ('                else (ANALYSIS_ROOT / "llmrouterbench" / ("results" + _SUFFIX) / PANEL.replace("llmrouterbench-", "")))\n',
   '                else (ANALYSIS_ROOT / "llmrouterbench" / ("results" + _SUFFIX) / PANEL.replace("llmrouterbench-", "")))\n'
   '# minilm/ study: an explicit (absolute) results root overrides the rule above; REP_NAME is the file stem of the\n'
   '# representation (the probe embeddings are stored as <REP_NAME>__<variant>.npy); E4_OUT overrides the output\n'
   '# folder of the probe scripts that live under results/<probe>/.  Unset => behaviour unchanged.\n'
   'if os.environ.get("E4_RESULTS_ROOT"):\n    RESULTS_ROOT = Path(os.environ["E4_RESULTS_ROOT"])\n'
   'REP_NAME = REP.split("/")[-1].split("@")[0]\n'
   'OUT_OVERRIDE = Path(os.environ["E4_OUT"]) if os.environ.get("E4_OUT") else None\n'),
  ('"llmrouterbench-performance-cost": 10, "llmrouterbench-performance": 15}[PANEL]',
   '"llmrouterbench-performance-cost": 10, "llmrouterbench-performance-cost-le512": 9, "llmrouterbench-performance": 15}[PANEL]'),
  ('{"llmrouterbench-performance-cost": "qwen3-235b-a22b-2507", "llmrouterbench-performance": "Qwen3-8B"}[PANEL]',
   '{"llmrouterbench-performance-cost": "qwen3-235b-a22b-2507", "llmrouterbench-performance-cost-le512": "qwen3-235b-a22b-2507", "llmrouterbench-performance": "Qwen3-8B"}[PANEL]')],
 "scripts/embed_gpu.py": [
  ('    "gte-qwen2-7b": (str(MODELS_DIR / "gte-Qwen2-7B-instruct"), 8192, None, "bfloat16", 4),\n',
   '    "gte-qwen2-7b": (str(MODELS_DIR / "gte-Qwen2-7B-instruct"), 8192, None, "bfloat16", 4),\n'
   '    # minilm/ study (PI, 2026-09-16): all-MiniLM-L12-v2 at its 512-token architectural window (the release config\n'
   '    # says 128); rows above 512 BERT tokens are excluded by the panel, so nothing is truncated.  fp32, unnormalised\n'
   '    # (the model\'s own Normalize module makes the vectors unit norm).\n'
   '    "minilm-l12": (str(MODELS_DIR / "all-MiniLM-L12-v2__3392ab41"), 512, None, None, 256),\n')],
 "scripts/exp16_hellaswag_check.py": [
  ("from rb.reps import find_rep, load_rep  # noqa: E402\n", "from rb.reps import find_rep, load_rep  # noqa: E402\nfrom rb.panelctx import PANEL, REP, RESULTS_ROOT  # noqa: E402\n"),
  ('    panel = load_panel("routerbench-0shot"); spec = ' + FR, "    panel = load_panel(PANEL); spec = find_rep(PANEL, REP)"),
  ('    d = ANALYSIS_ROOT / "results" / "outcome_locality" / EXP;', '    d = RESULTS_ROOT / "outcome_locality" / EXP;')],
 "scripts/exp21_cross_format_math.py": [
  ("from rb.reps import find_rep, load_rep  # noqa: E402\n", "from rb.reps import find_rep, load_rep  # noqa: E402\nfrom rb.panelctx import PANEL, REP, RESULTS_ROOT  # noqa: E402\n"),
  ('OUT = ANALYSIS_ROOT / "results" / "geometry_tests" / EXP', 'OUT = RESULTS_ROOT / "geometry_tests" / EXP'),
  ('    panel = load_panel("routerbench-0shot"); n = panel.n', "    panel = load_panel(PANEL); n = panel.n"),
  (FR, "find_rep(PANEL, REP)"),
  ('find_rep("routerbench-0shot", "computed/hash-bow@prompt")', 'find_rep(PANEL, "computed/hash-bow@prompt")')],
 "scripts/exp33_llm_topics.py": [
  ('    assert T["rows"] == n\n    labels = {"released": np.array([str(t) for t in topic_str(panel)], dtype=object),\n              "llm": np.array(T["topic"], dtype=object)}\n',
   '    llm_topic = np.array(T["topic"], dtype=object)\n'
   '    if panel.keep is not None and T["rows"] == len(panel.keep):\n'
   '        llm_topic = llm_topic[panel.keep]          # labels built on the full release; row-filtered panel (minilm/)\n'
   '    assert len(llm_topic) == n\n'
   '    labels = {"released": np.array([str(t) for t in topic_str(panel)], dtype=object),\n              "llm": llm_topic}\n')],
 "results/hard_examples/analyse.py": [
  ("from rb.reps import find_rep, load_rep  # noqa: E402\n", f"from rb.reps import find_rep, load_rep  # noqa: E402\n{IMP_FULL}  # noqa: E402\n"),
  ('OUT = HERE / "results"; LAMS', 'OUT = OUT_OVERRIDE or HERE / "results"; LAMS'),
  (FR, "find_rep(PANEL, REP)"), ('f"gte-qwen2-7b__{variant}.npy"', 'f"{REP_NAME}__{variant}.npy"')],
 "results/hard_examples/embed_hard_examples.py": [
  ("from embed_gpu import run_st  # noqa: E402\n", "from embed_gpu import run_st  # noqa: E402\nfrom rb.panelctx import REP_NAME  # noqa: E402\n"),
  ('run_st("gte-qwen2-7b", texts, HERE / "embeddings" / f"gte-qwen2-7b__{variant}.npy"', 'run_st(REP_NAME, texts, HERE / "embeddings" / f"{REP_NAME}__{variant}.npy"')],
 "results/format_invariance/analyse.py": [
  ("from rb.reps import find_rep, load_rep  # noqa: E402\n", f"from rb.reps import find_rep, load_rep  # noqa: E402\n{IMP_FULL}  # noqa: E402\n"),
  ('OUT = HERE / "results"; FMTS', 'OUT = OUT_OVERRIDE or HERE / "results"; FMTS'),
  (FR, "find_rep(PANEL, REP)"), ('f"gte-qwen2-7b__{f}.npy"', 'f"{REP_NAME}__{f}.npy"')],
 "results/format_invariance/embed_format_invariance.py": [
  ("from embed_gpu import run_st  # noqa: E402\n", "from embed_gpu import run_st  # noqa: E402\nfrom rb.panelctx import REP_NAME  # noqa: E402\n"),
  ('run_st("gte-qwen2-7b", texts, HERE / "embeddings" / f"gte-qwen2-7b__{v}.npy"', 'run_st(REP_NAME, texts, HERE / "embeddings" / f"{REP_NAME}__{v}.npy"')],
 "results/format_invariance/twin_vs_panel.py": [
  ("from rb.reps import find_rep, load_rep\n", f"from rb.reps import find_rep, load_rep\n{IMP_FULL}\n"),
  (FR, "find_rep(PANEL, REP)"), ('f"gte-qwen2-7b__{f}.npy"', 'f"{REP_NAME}__{f}.npy"'),
  ('(HERE / "results" / "twin_vs_panel.md")', '((OUT_OVERRIDE or HERE / "results") / "twin_vs_panel.md")')],
 "results/technique_twins/analyse.py": [
  ("from rb.reps import find_rep, load_rep  # noqa: E402\n", f"from rb.reps import find_rep, load_rep  # noqa: E402\n{IMP_FULL}  # noqa: E402\n"),
  ('HERE = Path(__file__).resolve().parent; OUT = HERE / "results"\n', 'HERE = Path(__file__).resolve().parent; OUT = OUT_OVERRIDE or HERE / "results"\n'),
  (FR, "find_rep(PANEL, REP)"), ('"gte-qwen2-7b__raw.npy"', 'f"{REP_NAME}__raw.npy"')],
 "results/technique_twins/embed_technique_twins.py": [
  ("from embed_gpu import run_st  # noqa: E402\n", "from embed_gpu import run_st  # noqa: E402\nfrom rb.panelctx import REP_NAME  # noqa: E402\n"),
  ('run_st("gte-qwen2-7b", [it["text"] for it in items], HERE / "embeddings" / "gte-qwen2-7b__raw.npy"', 'run_st(REP_NAME, [it["text"] for it in items], HERE / "embeddings" / f"{REP_NAME}__raw.npy"')],
 "results/technique_twins/mates_vs_panel.py": [
  ("from rb.reps import find_rep, load_rep\n", f"from rb.reps import find_rep, load_rep\n{IMP_FULL}\n"),
  (FR, "find_rep(PANEL, REP)"), ('"gte-qwen2-7b__raw.npy"', 'f"{REP_NAME}__raw.npy"'),
  ('HERE / "results" / "mates_vs_panel.', '(OUT_OVERRIDE or HERE / "results") / "mates_vs_panel.')],
 "results/deformatting/analyse.py": [
  ("from rb.reps import find_rep, load_rep  # noqa: E402\n", f"from rb.reps import find_rep, load_rep  # noqa: E402\n{IMP_FULL}  # noqa: E402\n"),
  ("OUT = HERE; KS", "OUT = OUT_OVERRIDE or HERE; KS"), (FR, "find_rep(PANEL, REP)"), ('PANEL_DIR / "gte-qwen2-7b.npy"', 'PANEL_DIR / f"{REP_NAME}.npy"')],
 "results/deformatting/build_and_embed_core.py": [
  ("from deformat import PANEL_DIR, deformat  # noqa: E402\n", "from deformat import PANEL_DIR, deformat  # noqa: E402\nfrom rb.panelctx import REP_NAME  # noqa: E402\n"),
  ('run_st("gte-qwen2-7b", core, PANEL_DIR / "gte-qwen2-7b.npy",', 'run_st(REP_NAME, core, PANEL_DIR / f"{REP_NAME}.npy",')],
 "results/deformatting/family_confusion.py": [
  ("from deformat import PANEL_DIR\n", f"from deformat import PANEL_DIR\n{IMP_FULL}\n"),
  (FR, "find_rep(PANEL, REP)"), ('PANEL_DIR / "gte-qwen2-7b.npy"', 'PANEL_DIR / f"{REP_NAME}.npy"'),
  ('(HERE / "family_confusion.md")', '((OUT_OVERRIDE or HERE) / "family_confusion.md")'),
  ('open(HERE / "family_confusion.json", "w")', 'open((OUT_OVERRIDE or HERE) / "family_confusion.json", "w")')],
}
MARK = {"src/rb/data.py": "performance-cost-le512", "src/rb/panelctx.py": "OUT_OVERRIDE", "scripts/embed_gpu.py": '"minilm-l12"', "scripts/exp33_llm_topics.py": "labels built on the full release"}

def main() -> None:
    for rel, edits in EDITS.items():
        p = ROOT / rel; s = p.read_text()
        if MARK.get(rel, "rb.panelctx") in s:
            print("skip (already ported)", rel); continue
        bk = BK / rel; bk.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, bk)
        for old, new in edits:
            k = s.count(old)
            assert k >= 1, (rel, old[:60]); assert k == 1 or rel.endswith("mates_vs_panel.py"), (rel, old[:60], k)
            s = s.replace(old, new)
        p.write_text(s); print("ported", rel, len(edits), "edits")

if __name__ == "__main__":
    main()
