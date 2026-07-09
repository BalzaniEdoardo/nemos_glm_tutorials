# Installation

These tutorials run on **Python 3.12–3.14**. The steps below get you a fresh
environment with everything needed to run the notebooks locally.

First, clone the repository:

```bash
git clone https://github.com/BalzaniEdoardo/nemos_glm_tutorials.git
cd nemos_glm_tutorials
```

Then pick your package manager:

::::{tab-set}

:::{tab-item} uv
[`uv`](https://docs.astral.sh/uv/) creates and manages the virtual environment
for you — one command installs the project with the `notebooks` extra into a
fresh `.venv`:

```bash
uv sync --extra notebooks
```

Launch JupyterLab inside that environment:

```bash
uv run jupyter lab
```
:::

:::{tab-item} conda
Create and activate a fresh environment, then install with the `notebooks`
extra:

```bash
conda create -n nemos-tutorials python=3.12
conda activate nemos-tutorials

pip install -e ".[notebooks]"
```

Launch JupyterLab:

```bash
jupyter lab
```
:::

::::

## Running a tutorial

The tutorial notebooks live on this website. To run one locally:

1. Open the tutorial page you want (browse the collections from the
   [home page](index.md)).
2. At the top of the page, in the **"Run this tutorial yourself"** box, click the
   **Jupyter notebook (.ipynb)** download link to save the `.ipynb`.
3. Launch JupyterLab in the environment you created above and open the downloaded
   notebook:

   ```bash
   # uv
   uv run jupyter lab

   # conda (with the env activated)
   jupyter lab
   ```

Every notebook downloads clean (no stored outputs), so running it top to bottom
reproduces the figures from scratch.