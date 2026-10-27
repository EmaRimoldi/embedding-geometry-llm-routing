# Figure 1 in motion

The README animation reuses **every original drawing object** from Figure 1 of
[Embedding Geometry Shapes LLM Routing](https://arxiv.org/abs/2610.19198).
The source is Overleaf's `aistats_figures/intro_pipeline.pdf` at snapshot
`6e3bea88ec5d65de3587c0719025702e110ac3e0`. The unchanged PDF is included here;
font outlines, colors, points, decision boundary, arrows and model symbols are
preserved. The 137 vector objects are separated into 30 named animation layers.
[provenance.json](provenance.json) records the source checksum and full layer map.

![Final animation state](routing-poster.png)

## Play it

The [GIF](routing-demo.gif) plays directly in the GitHub README.
For pause/play, replay, prompt selection and timeline controls, open the cloned
repository's `docs/demo/index.html` in a browser. It is self-contained and works
offline, including from a `file:` URL. It honors reduced-motion preferences.

On macOS, from the repository root:

```bash
open docs/demo/index.html
```

The prompts, two-dimensional layout and model IDs illustrate the paper's routing
pipeline. This explanation does not run a trained router or an LLM in the browser.
The same schematic model 2 is selected for the illustrative prompts.

## What the counters measure

The numbers come from the supplied `results/evidence/baselines/_summary_reduced/scores.csv`:
RouterBench, held-out-family evaluation, **all 13 frozen cost penalties**. At each
penalty, results pool held-out queries; the displayed values are the arithmetic
mean across penalties. The reference is **Best single model**, chosen using source
outcomes at each penalty, compared with **L21 Family Adaptation**.

| Metric | Source-selected single model | L21 Family Adaptation |
| --- | ---: | ---: |
| Benchmark reward | 0.7104450437 | 0.7463264037 |
| Inference cost, USD/query | 0.002071782337 | 0.001548069657 |
| Mean utility | 0.6618000328 | 0.7156970263 |

Thus the quality counter increases by **3.59 percentage points** and the cost
counter decreases by **25.3%**. These are aggregate comparisons, not measurements
for the displayed prompt. Reward includes graded tasks, so it is not uniformly
binary accuracy. Cost is the recorded inference cost of the chosen candidate
model; it does not add encoder/router overhead. A cost penalty does not imply that
every prompt or every operating point improves both metrics simultaneously.

[demo-data.json](demo-data.json) retains full precision, the exact metric scope,
selectors and source-file hash. `scripts/build_readme_demo.py` recomputes these
values from the frozen records; they are not hand-entered animation targets.

## Rebuild the animation

These optional authoring dependencies are separate from the experiment environment.
The renderer uses Playwright 1.62.1; Chrome can alternatively be selected with
`DEMO_BROWSER=chrome` if installed.

```bash
python3.11 -m venv .venv-demo
.venv-demo/bin/pip install -r environment/demo-requirements.lock
npm install --prefix .cache/readme-demo playwright@1.62.1
.cache/readme-demo/node_modules/.bin/playwright install chromium

.venv-demo/bin/python scripts/build_readme_demo.py
NODE_PATH=.cache/readme-demo/node_modules node scripts/render_readme_demo.cjs
.venv-demo/bin/python scripts/build_readme_demo.py --gif
```

The builder decomposes the original PDF and embeds its vectors and measured values
in the offline page. The renderer captures 128 deterministic timeline samples over
12.8 seconds and checks the controls, final values, reduced-motion behavior and
layouts at 320, 768 and 1200 pixels. Frame PNGs stay under `.cache/readme-demo/`;
the [browser-check report](browser-check.json), compressed GIF and final-state
poster are exported here.
Renderer/font versions can affect raster bytes; the original vectors and numerical
inputs have explicit hashes.

After changing the demo or README, update the repository's file manifest and export:

```bash
python scripts/build_submission.py
python scripts/verify_release.py
```
