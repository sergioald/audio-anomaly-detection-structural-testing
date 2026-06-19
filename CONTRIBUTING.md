# Contributing

Thank you for your interest in improving this repository.

This is a research-software companion repository for an audio-based anomaly-detection workflow in large-scale structural testing. Contributions should preserve the public/private data boundary and keep the lightweight CI path fast and reproducible.

## Development setup

Create an environment and install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the pre-commit hooks locally:

```bash
pre-commit install
```

Run the same checks used by CI:

```bash
pre-commit run --all-files
pytest
```

## Optional dependencies

The default development path intentionally avoids heavy dependencies. Install optional extras only when needed:

```bash
python -m pip install -e ".[deep-learning]"
python -m pip install -e ".[wst]"
```

Use `deep-learning` for TensorFlow/Keras model operations and `wst` for raw-audio/WST processing.

## Testing policy

The core test suite should remain lightweight. It should:

- use small synthetic arrays;
- avoid downloading the full Zenodo dataset;
- avoid training the CAE;
- avoid requiring a GPU;
- avoid requiring TensorFlow/Keras, SciPy, or Kymatio in the base CI path.

Tests for heavy workflows are welcome, but they should be clearly separated from the default CI path.

## Data and confidentiality policy

Do not commit:

- raw FastBlade audio;
- confidential facility data;
- proprietary control-system information;
- private paths, names, or operational records;
- large downloaded datasets.

The repository should remain focused on the public processed Zenodo dataset and reusable local workflows for user-provided data.

## Documentation

Please update documentation when changing user-facing behaviour. Good places to update are:

- `README.md` for the main public workflow;
- `docs/reproducibility.md` for commands;
- `docs/scientific_validation.md` for testing and validation scope;
- `docs/new_data_workflow.md` for raw-audio workflows;
- `docs/confidentiality_statement.md` for public/private data boundaries.

## Code style

The project uses `ruff` and `pre-commit`.

Before committing, run:

```bash
ruff check --fix .
ruff format .
pre-commit run --all-files
pytest
```

## Pull request checklist

Before opening a pull request, check that:

- tests pass locally;
- pre-commit passes locally;
- no private or large data files are committed;
- documentation is updated when needed;
- scientific claims remain appropriately qualified.
