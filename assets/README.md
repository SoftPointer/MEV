# README visual assets

## Project icon

[mev-icon.png](mev-icon.png) is the existing [project-page favicon](https://mev-dataset.github.io/fig/favicon.png), downloaded on 2026-10-27. The original 128 × 128 PNG and its transparency are preserved. It is displayed inline at 48 × 48 pixels immediately before the left-aligned title, `MEV: Multi-Event Video Dataset`.

Source SHA-256: `7007e211a1a448206bc563fdcb4385eccfbee3702692a60ce51445cada8cccdb`.

The conference and resource links share one centered row of compact badges below the title. The full paper title remains in the citation metadata.

## Header badges

The header uses standard [Shields.io static badges](https://shields.io/badges/static-badge) in the native 20-pixel `flat` style, with matching slate label fields. From left to right, the value fields are blue for Project, gold for Dataset, teal for MEV-Eval, rose for Citation, and purple for NeurIPS 2026. The SVGs are served directly by Shields.io. Each badge remains a separate link, with consistent spacing, so the centered row can wrap on narrow screens.

## Paper figures

These figures come from the camera-ready source of *MEV: A Multi-Event Video Dataset for Long-Take Generation*. They were rendered from the original PDFs on 2026-10-27; no new examples or measurements were generated.

| GitHub asset | Paper source under `camera_ready/src/figures/` | Purpose | Export size |
|---|---|---|---|
| [overview.png](overview.png) | `overview.pdf` | Main README teaser; ordered events across diverse continuous videos | 2400 × 1308 |
| [annotation.png](annotation.png) | `caption.pdf` | Global/event annotation structure | 2400 × 1120 |
| [statistics.png](statistics.png) | `stats2_cropped.pdf` | Dataset statistics | 2400 × 1687 |

The overview links to the project page and its video. The annotation scheme (Figure 2 in the paper) is displayed directly below the dataset download instructions. The statistics figure is available here as a supporting asset. All figures use a white background for readability in GitHub light and dark themes; click an image to inspect the full resolution.

## Regenerate from the paper source

With Poppler installed, run from the public repository root. Set `MEV_FIGURE_DIR` to the paper's figure directory:

```bash
pdftoppm -f 1 -singlefile -scale-to 2400 -png \
  "$MEV_FIGURE_DIR/overview.pdf" assets/overview
pdftoppm -f 1 -singlefile -scale-to 2400 -H 1120 -png \
  "$MEV_FIGURE_DIR/caption.pdf" assets/annotation
pdftoppm -f 1 -singlefile -scale-to 2400 -png \
  "$MEV_FIGURE_DIR/stats2_cropped.pdf" assets/statistics
```

The annotation render omits the unused bottom margin and a stray footer mark in the source page; the figure content is preserved. The source PDFs are not modified.

Source SHA-256 hashes:

```text
overview.pdf        5184e33e7916580de2a071a7f672b084274a915f3286c5059158fc71d2d0d5e7
caption.pdf         42c2f83be1bca2842a4e75d0270497a3c47e1f480ee668c5afc4a0c290f7e44c
stats2_cropped.pdf  73487f9b7eca1527f593e70fa8a26b69dd9ff7588bbf96509ec4e002c4fa2244
```

## Optional future assets

The current GitHub README uses the overview and annotation figures. A short compressed GIF showing one real multi-event sequence could be added later if an inline motion preview is useful. The full project video should remain on the project page or a release host rather than being checked into Git. A formal paper button can be added when a public paper URL is available.

See [DATA_LICENSE.md](../DATA_LICENSE.md) for figure and embedded-frame terms.
