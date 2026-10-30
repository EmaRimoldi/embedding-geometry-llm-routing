#!/usr/bin/env python3
"""Decompose the original Figure 1 vectors and build the offline README demo.

Optional authoring dependencies: environment/demo-requirements.lock.
No model is executed and no experimental evidence is modified by this script.
"""
import argparse
import csv
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import statistics
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import fitz

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / "docs/demo"
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


def cached_font():
    """Fetch only the optional illustration font, at a pinned revision/checksum."""
    record = json.loads((ROOT / "manifests/readme_font.json").read_text())
    asset = record["font"]
    destination = ROOT / ".cache/readme-demo/font-assets/Inter-Variable.ttf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".ttf", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--output", str(temporary), asset["url"]], check=True)
            if sha(temporary) != asset["sha256"] or temporary.stat().st_size != asset["bytes"]:
                raise ValueError("Downloaded illustration font differs from the pinned asset")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if sha(destination) != asset["sha256"] or destination.stat().st_size != asset["bytes"]:
        raise ValueError("Cached illustration font differs from its pinned checksum")
    license_path = ROOT.parent / record["license"]["path"]
    if sha(license_path) != record["license"]["sha256"]:
        raise ValueError("Illustration font license differs from its pinned checksum")
    return destination


def build_comparison(rows, relative):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    font_source = cached_font()
    font_cache = ROOT / ".cache/readme-demo/fonts"
    font_cache.mkdir(parents=True, exist_ok=True)
    fonts = {}
    for name, weight in [("Regular", 400), ("SemiBold", 600)]:
        path = font_cache / ("Inter-" + name + ".ttf")
        with TTFont(font_source) as variable:
            instance = instantiateVariableFont(variable, {"opsz": 14, "wght": weight},
                                               inplace=False, updateFontNames=True)
            instance.save(path)
        font_manager.fontManager.addfont(str(path))
        fonts[name] = font_manager.FontProperties(fname=path)

    panels = []
    for dataset, label in [("routerbench-0shot", "RouterBench"), ("sprout-sixfam", "SPROUT")]:
        curves = []
        for encoder, name, color, marker, style in [
            ("gte", "GTE semantic embedding", "#0072b2", "o", "--"),
            ("reader", "Qwen L21 hidden state", "#c43b3b", "s", "-"),
        ]:
            subset = sorted([r for r in rows if r["dataset"] == dataset
                             and r["protocol"] == "ood" and r["encoder"] == encoder
                             and r["method"] == "kNN-100"],
                            key=lambda r: float(r["lambda_value"]))
            assert len(subset) == 13 and all(r["complete"] == "True" for r in subset)
            points = [{key: float(r[field]) for key, field in
                       [("penalty", "lambda_value"), ("utility", "utility"),
                        ("reward", "reward"), ("cost", "cost")]}
                      for r in subset]
            assert len({p["penalty"] for p in points}) == 13
            assert all(math.isclose(p["utility"], p["reward"] - p["penalty"] * p["cost"],
                                    rel_tol=1e-10, abs_tol=1e-12) for p in points)
            curves.append(dict(encoder=encoder, label=name, color=color,
                               marker=marker, linestyle=style, points=points,
                               mean_utility=statistics.mean(p["utility"] for p in points)))
        assert [p["penalty"] for p in curves[0]["points"]] == [p["penalty"] for p in curves[1]["points"]]
        panels.append(dict(dataset=dataset, label=label, curves=curves))

    data = dict(method="kNN-100", protocol="held-out-family",
                scope="Query-pooled held-out utility at every frozen penalty; identical router and cohorts across embeddings. Lines connect evaluated points without smoothing.",
                utility="reward - penalty * cost", x_axis="symlog, linear within [0, 1]",
                evidence=relative, evidence_sha256=sha(ROOT / relative), panels=panels)
    (DEST / "embedding-comparison-data.json").write_text(json.dumps(data, indent=2) + "\n")
    rc = {"font.family": fonts["Regular"].get_name(), "font.size": 11.5, "axes.spines.top": False,
          "axes.spines.right": False, "axes.labelcolor": "#4b5563",
          "text.color": "#26313b", "xtick.color": "#586572", "ytick.color": "#586572",
          "axes.edgecolor": "#c4cdd5", "svg.hashsalt": "routing-embedding-comparison",
          "svg.fonttype": "path", "savefig.facecolor": "white"}
    with plt.rc_context(rc):
        for stacked in [False, True]:
            fig, axes = plt.subplots(2 if stacked else 1, 1 if stacked else 2,
                                     figsize=(6.2, 7.3) if stacked else (12, 4.4),
                                     sharey=True)
            if stacked:
                fig.subplots_adjust(left=.13, right=.96, top=.73, bottom=.085, hspace=.55)
            else:
                fig.subplots_adjust(left=.075, right=.98, top=.71, bottom=.19, wspace=.17)
            fig.text(.075 if not stacked else .13, .935, "Same router. Different embeddings.",
                     fontsize=20 if not stacked else 17, fontproperties=fonts["SemiBold"])
            for ax, panel in zip(axes.flat, panels):
                for curve in panel["curves"]:
                    ax.plot([p["penalty"] for p in curve["points"]],
                            [p["utility"] for p in curve["points"]], label=curve["label"],
                            color=curve["color"], marker=curve["marker"],
                            linestyle=curve["linestyle"], linewidth=2.2, markersize=4.3)
                ax.set_title(panel["label"], loc="left", fontsize=13.5,
                             fontproperties=fonts["SemiBold"], pad=12)
                ax.set_xscale("symlog", linthresh=1, linscale=.7)
                ax.set_xlim(-.1, 650)
                ax.set_xticks([0, 1, 10, 100, 500], labels=["0", "1", "10", "100", "500"])
                ax.minorticks_off()
                ax.set_ylim(.48, .855)
                ax.set_yticks([.5, .6, .7, .8])
                ax.set_xlabel("Cost penalty λ (symlog scale)", labelpad=10)
                ax.grid(axis="y", color="#e6ebef", linewidth=.8)
                ax.set_axisbelow(True)
                ax.tick_params(length=0, pad=7)
                if stacked:
                    ax.set_ylabel("Utility", labelpad=10)
            if not stacked:
                axes.flat[0].set_ylabel("Utility", labelpad=10)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper left",
                       bbox_to_anchor=(.066 if not stacked else .11, .865 if not stacked else .90),
                       ncol=2 if not stacked else 1, frameon=False, fontsize=11.5,
                       handlelength=2.5, columnspacing=2)
            name = "embedding-comparison-stacked" if stacked else "embedding-comparison"
            svg_path = DEST / (name + ".svg")
            fig.savefig(svg_path, metadata={"Date": None, "Creator": "Matplotlib " + matplotlib.__version__})
            plot = ET.fromstring(svg_path.read_text())
            plot.set("role", "img")
            plot.set("aria-label", "GTE in blue and Qwen L21 in red: held-out kNN-100 utility on RouterBench and SPROUT across all 13 cost penalties")
            serialized = ET.tostring(plot, encoding="unicode")
            svg_path.write_text("\n".join(line.rstrip() for line in serialized.splitlines()) + "\n")
            if not stacked:
                fig.savefig(DEST / "embedding-comparison.png", dpi=600,
                            metadata={"Software": "Matplotlib " + matplotlib.__version__})
            plt.close(fig)
    return data


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

    relative = "results/evidence/baselines/_summary_reduced/scores.csv"
    with (ROOT / relative).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    comparison = build_comparison(rows, relative)
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
        "animation": {"duration_seconds": 10.8, "fps": 10},
        "embedding_comparison": {
            p["label"]: {c["encoder"]: c["mean_utility"] for c in p["curves"]}
            for p in comparison["panels"]},
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
        "comparison_plot": "All 13 penalties per benchmark, fixed kNN-100, GTE versus L21; embedding-comparison-data.json",
        "plot_renderer": "Matplotlib " + version("matplotlib"),
        "presentation": {"plot_font": "Inter", "plot_font_sha256": sha(cached_font()),
                         "png_dpi": 600, "png_dimensions": [7200, 2640],
                         "animation_pixel_ratio": 2},
    }
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"layers": len(GROUPS), "original_objects_retained": len(indices),
                      "reward_gain_pp": data["reward_gain_pp"],
                      "cost_reduction_percent": data["cost_reduction_percent"]}, indent=2))


def make_gif():
    from PIL import Image
    frames = sorted((ROOT / ".cache/readme-demo/frames").glob("frame-*.png"))
    animation = json.loads((DEST / "demo-data.json").read_text())["animation"]
    expected = round(animation["duration_seconds"] * animation["fps"])
    assert len(frames) == expected, "Run scripts/render_readme_demo.cjs first"
    with Image.open(DEST / "routing-poster.png") as poster:
        palette = poster.convert("RGB").quantize(colors=256)
    images = []
    for path in frames:
        with Image.open(path) as frame:
            images.append(frame.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE))
    target = DEST / "routing-demo.gif"
    images[0].save(target, save_all=True, append_images=images[1:], duration=round(1000 / animation["fps"]),
                   loop=0, optimize=True, disposal=1)
    with Image.open(target) as gif:
        duration = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            duration += gif.info["duration"]
        assert duration == round(1000 * animation["duration_seconds"]) and gif.info["loop"] == 0
        assert target.stat().st_size < 5_000_000
        print(json.dumps({"gif_bytes": target.stat().st_size,
                          "frames": gif.n_frames, "duration_ms": duration}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", action="store_true", help="Encode previously rendered frames")
    args = parser.parse_args()
    make_gif() if args.gif else build()
