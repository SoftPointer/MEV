# Maintainer release notes

Code repository: [SoftPointer/MEV](https://github.com/SoftPointer/MEV). The project website and Hugging Face dataset are maintained separately; publishing code does not update either service.

## Publishing layout

This directory is the **root of the MEV code repository**. In the research workspace it is named `repo_code/`; synchronize its contents, including dotfiles, with the code repository root. The `.github/` workflow and templates must be at that root for GitHub to discover them.

The surrounding research workspace contains data preparation, annotation work, and local experiments that are outside this package. Use an isolated checkout of `SoftPointer/MEV` to keep its history separate from the research repository.

GitHub About description:

> [NeurIPS 2026] MEV: A Multi-Event Video Dataset for Long-Take Generation

Website: `https://mev-dataset.github.io/`

Topics: `video-dataset`, `text-to-video`, `multi-event`, `video-generation`, `video-captioning`, `video-evaluation`, `neurips-2026`.

Use `Update` as the commit message for routine repository maintenance.

## Prepared files

- English README with a compact title and inline icon, a centered row of standard Shields.io conference and resource badges, three project highlights, visible overview and annotation figures, and evaluator quickstarts. Citations retain the full paper title.
- Three PNGs rendered from the camera-ready figures, with source hashes and regeneration commands in [assets/README.md](../assets/README.md).
- [Dataset guide](DATASET.md) and [reproducibility scope](REPRODUCIBILITY.md).
- MIT code license, separate data/media notice, BibTeX, and `CITATION.cff`.
- `.gitignore`, `.gitattributes`, `.env.example`, contribution instructions, issue/PR templates, and a GitHub Actions workflow using the existing offline tests.

## Link and metadata checks on 2026-10-27

The project page and Hugging Face repository were reachable. The inspected dataset revision was `1e9460d5909116807dd45345b174fb91e9244553`. Download paths in the guide were checked against that revision's file inventory.

At inspection time, the project page had empty, disabled `Paper`, `Code`, and `Dataset` button links. For a separately requested website update, the intended destinations are `https://huggingface.co/datasets/MultiEventVideo/MEV` for `Dataset` and `https://github.com/SoftPointer/MEV` for `Code`; add `Paper` when its public URL is available. These changes belong to the separate project-page repository.

The README and `CITATION.cff` now cite the arXiv record [2610.38241](https://arxiv.org/abs/2610.38241) (title, author order, and year verified against the arXiv listing). The project page's existing BibTeX still says “submission”; updating it belongs to the separate project-page repository.

The Hub dataset card contains an older metadata-only distribution sentence, while its newer `LICENSE` describes the downloadable video package. This GitHub documentation follows the newer license and verified file inventory. Synchronizing the Hub card is a separate maintenance task.

## Verified locally

- All 14 existing tests passed on Python 3.8 and Python 3.9 without calling Gemini.
- The README's cached-response example produced the expected summary and CSV.
- The shell launcher passed `bash -n`.
- Relative documentation links resolve, and YAML/JSON files parse successfully.
- `CITATION.cff` validates against the official CFF 1.2.0 schema.
- The three exported figures were visually inspected. Hosted test results are recorded in [GitHub Actions](https://github.com/SoftPointer/MEV/actions/workflows/tests.yml).

## Release maintenance

1. Synchronize the prepared package with the code repository when publication is requested.
2. Update the project-page buttons and add the public paper link only when a website update is separately requested.
3. Verify the GitHub Actions run for the published commit.
4. Keep the missing reproducibility inputs visible in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Add exact UUID partitions, prompt draws, and seeds only when recovered and verified.

No model-weight hosting, model download instructions, or promised weight release is part of this package.
