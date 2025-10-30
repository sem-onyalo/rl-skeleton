# rlskeleton

Skeleton code for RL applications.

## Setup

1. Clone repository
    ```bash
    git clone https://github.com/sem-onyalo/rl-skeleton.git
    cd rl-skeleton
    ```

1. Install uv
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

1. Setup environment
    ```bash
    pip install pre-commit
    pre-commit install
    uv sync
    ```

1. Setup for development
    ```bash
    uv pip install -e .
    ```

## Pre-Commit Checks

```bash
uvx ruff format && uvx ruff check
```

## Tests

```bash
uv run pytest
```
