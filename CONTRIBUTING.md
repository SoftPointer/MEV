# Contributing

For evaluator bugs, include the command, Python version, expected result, observed result, and a minimal record or cached judge response when possible. Remove API keys and private paths from logs. Dataset questions belong on the [Hugging Face discussion page](https://huggingface.co/datasets/MultiEventVideo/MEV/discussions).

Keep changes focused. Describe any change to a metric definition, judge prompt, or temporal-region assignment explicitly, since these can affect comparability with existing results.

Run the existing offline checks for evaluator changes:

```bash
cd mev_eval
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Use a stub judge or cached response when adding coverage. Routine tests should not require credentials or make paid API calls. For documentation-only edits, check links and example commands relevant to the change.

Code contributions are made under the repository's MIT license. Dataset and media terms are described separately in [DATA_LICENSE.md](DATA_LICENSE.md).
