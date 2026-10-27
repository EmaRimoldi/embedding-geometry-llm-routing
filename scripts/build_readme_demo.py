#!/usr/bin/env python3
"""Decompose the original Figure 1 vectors and build the offline README demo.

Optional authoring dependencies: environment/demo-requirements.lock.
No model is executed and no experimental evidence is modified by this script.
"""
import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import fitz

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/demo"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# Indices of the original PDF-to-SVG drawing objects, including font outlines.
# Every object is retained exactly once; no icon or decision boundary is redrawn.
GROUPS = [
    ("background", [0]), ("labels", list(range(1, 71))),
    ("prompt", [71, 72]),
    *[(f"prompt-line-{i}", [73 + i]) for i in range(3)],
    ("prompt-arrow", [76, 77]), ("encoder", [78, 79]),
    ("encoder-arrow", [80, 81]), ("vector-brackets", [82, 99]),
    *[(f"vector-cell-{i}", [83 + 2 * i, 84 + 2 * i]) for i in range(8)],
    ("embedding-arrow", [100, 101]),
    ("classifier", [102, 103, 104]),
    ("class-gray", list(range(105, 111))),
    ("class-blue", list(range(111, 117))),
    ("query-star", [117]),
    ("other-routes", [118, 119, 122, 123]),
    ("selected-route", [120, 121]),
    ("model-one", [124, 125, 126]),
    ("model-two", [127, 128, 129]), ("selection-check", [130]),
    ("model-ellipsis", [131, 132, 133]),
    ("model-many", [134, 135, 136]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    source = DEST / "figure1-original.pdf"
    with fitz.open(source) as pdf:
        assert len(pdf) == 1 and abs(pdf[0].rect.width - 720) < .01
        tree = ET.fromstring(pdf[0].get_svg_image(text_as_path=True))
    artwork = list(tree)[1]
    children = list(artwork)
    indices = [i for _, group in GROUPS for i in group]
    assert len(children) == 137 and sorted(indices) == list(range(137))
    for child in children:
        artwork.remove(child)
    for name, group in GROUPS:
        layer = ET.SubElement(artwork, f"{{{SVG}}}g", {"id": name})
        for index in group:
            layer.append(children[index])
    tree.set("id", "routing-figure")
    tree.set("role", "img")
    tree.set("aria-labelledby", "figure-title figure-desc")
    title = ET.Element(f"{{{SVG}}}title", {"id": "figure-title"})
    title.text = "Figure 1: prompt, embedding, classification, model selection"
    desc = ET.Element(f"{{{SVG}}}desc", {"id": "figure-desc"})
    desc.text = "Original paper vectors, separated into layers for animation. Model 2 is schematic."
    tree.insert(0, title)
    tree.insert(1, desc)
    svg = ET.tostring(tree, encoding="unicode")
    (DEST / "figure1-components.svg").write_text(svg + "\n")

    relative = "results/evidence/baselines/_summary_reduced/scores.csv"
    with (ROOT / relative).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summaries = {}
    for key, encoder, method in [
        ("baseline", "gte", "Best single model"),
        ("routed", "reader", "Family Adaptation"),
    ]:
        subset = [r for r in rows if r["dataset"] == "routerbench-0shot"
                  and r["protocol"] == "ood" and r["encoder"] == encoder
                  and r["method"] == method]
        assert len(subset) == 13 and all(r["complete"] == "True" for r in subset)
        summaries[key] = {
            "encoder": encoder, "method": method,
            "penalties": sorted(float(r["lambda_value"]) for r in subset),
            **{metric: statistics.mean(float(r[metric]) for r in subset)
               for metric in ["reward", "cost", "utility"]},
        }
    assert summaries["baseline"]["penalties"] == summaries["routed"]["penalties"]
    baseline, routed = summaries["baseline"], summaries["routed"]
    assert routed["reward"] > baseline["reward"] and routed["cost"] < baseline["cost"]
    data = {
        "scope": "RouterBench held-out-family results, query-pooled per penalty, arithmetic mean across all 13 frozen penalties",
        "quality_label": "Answer quality (benchmark reward)",
        "baseline_label": "Source-selected single model",
        "routed_label": "L21 Family Adaptation",
        **summaries,
        "reward_gain_pp": 100 * (routed["reward"] - baseline["reward"]),
        "cost_reduction_percent": 100 * (1 - routed["cost"] / baseline["cost"]),
        "evidence": relative, "evidence_sha256": sha(ROOT / relative),
        "illustrative_only": "Prompts, 2D layout and model IDs illustrate the routing pipeline; they are not live model predictions or per-prompt cost/accuracy measurements.",
    }
    (DEST / "demo-data.json").write_text(json.dumps(data, indent=2) + "\n")
    template = (DEST / "template.html").read_text()
    assert template.count("__FIGURE_SVG__") == template.count("__DEMO_DATA__") == 1
    html = template.replace("__FIGURE_SVG__", svg).replace("__DEMO_DATA__", json.dumps(data))
    (DEST / "index.html").write_text(html)
    provenance = {
        "source": "Overleaf aistats_figures/intro_pipeline.pdf, Figure 1 in main.tex",
        "overleaf_snapshot": "6e3bea88ec5d65de3587c0719025702e110ac3e0",
        "source_pdf_sha256": sha(source),
        "vector_converter": "PyMuPDF " + version("PyMuPDF"),
        "source_objects": 137, "objects_retained": len(indices),
        "layers": {name: group for name, group in GROUPS},
        "metrics": data["scope"], "evidence_sha256": data["evidence_sha256"],
    }
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"layers": len(GROUPS), "original_objects_retained": len(indices),
                      "reward_gain_pp": data["reward_gain_pp"],
                      "cost_reduction_percent": data["cost_reduction_percent"]}, indent=2))


def make_gif():
    from PIL import Image
    frames = sorted((ROOT / ".cache/readme-demo/frames").glob("frame-*.png"))
    assert len(frames) == 128, "Run scripts/render_readme_demo.cjs first"
    with Image.open(DEST / "routing-poster.png") as poster:
        palette = poster.convert("RGB").quantize(colors=256)
    images = []
    for path in frames:
        with Image.open(path) as frame:
            images.append(frame.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE))
    target = DEST / "routing-demo.gif"
    images[0].save(target, save_all=True, append_images=images[1:], duration=100,
                   loop=0, optimize=True, disposal=1)
    with Image.open(target) as gif:
        duration = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            duration += gif.info["duration"]
        assert duration == 12800 and gif.info["loop"] == 0
        assert target.stat().st_size < 5_000_000
        print(json.dumps({"gif_bytes": target.stat().st_size,
                          "frames": gif.n_frames, "duration_ms": duration}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", action="store_true", help="Encode previously rendered frames")
    args = parser.parse_args()
    make_gif() if args.gif else build()
