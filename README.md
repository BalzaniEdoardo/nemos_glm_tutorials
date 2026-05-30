<p align="center">
  <img src="docs/_static/pynapple_icon.png" alt="pynapple" height="70">
  &nbsp;&nbsp;&nbsp;
  <img src="docs/_static/NeMoS_Icon_CMYK_Full.svg" alt="NeMoS" height="70">
</p>

# Pynapple & NeMoS tutorials

A collection of systems-neuroscience tutorials built around
[`pynapple`](https://pynapple.org/) for data handling and
[`NeMoS`](https://nemos.readthedocs.io/en/latest/index.html) for modeling. Each
collection reproduces a well-known course or paper, so you can see modern tools
applied to familiar material.

📖 **Read the tutorials:** https://balzaniedoardo.github.io/nemos_glm_tutorials/

## Tutorial collections

- **Data Science & Data Skills for Neuroscientists — J.W. Pillow (SfN 2016).** A
  faithful rebuild of Jonathan Pillow's
  [spike-train GLM short course](https://github.com/pillowlab/GLMspiketraintutorial_python),
  reproduced as a series of tutorials on fitting GLMs to retinal ganglion cell
  spike trains with `pynapple` and `NeMoS`.

## Installation

Requires Python 3.12–3.14. Clone the repo and install with the extra that matches
what you want to do:

```bash
git clone https://github.com/BalzaniEdoardo/nemos_glm_tutorials.git
cd nemos_glm_tutorials

# to run the tutorial notebooks locally
pip install -e ".[notebooks]"

# to build the documentation site
pip install -e ".[docs]"
```

## Running the notebooks

The tutorials are stored as [MyST-Markdown](https://mystmd.org/) notebooks
(`docs/tutorials/**/*.md`), paired to Jupyter via
[jupytext](https://jupytext.readthedocs.io/). Open the repository in JupyterLab
and the `.md` files open as executable notebooks:

```bash
jupyter lab
```

To render the MyST directives (admonitions, dropdowns, equation references) inside
JupyterLab, also install the
[`jupyterlab-myst`](https://github.com/jupyter-book/jupyterlab-myst) extension
(included in the `notebooks` extra).

## Building the docs locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
# then open docs/_build/html/index.html
```

The build executes every notebook, so the rendered site always reflects working
code. On every push to `main` it is built and deployed to GitHub Pages by the
[`docs` workflow](.github/workflows/docs.yml).

## Data

Datasets are downloaded on demand with [`pooch`](https://www.fatiando.org/pooch/)
through the `fetch_data` helper:

```python
from nemos_tutorials import fetch_data

paths = fetch_data("data_RGCs")  # -> {filename: local_path}
```

The retinal ganglion cell data are from
[Uzzell & Chichilnisky, 2004](https://pubmed.ncbi.nlm.nih.gov/15277596/). They are
provided for tutorial purposes only and should not be distributed or used for
publication without express permission from EJ Chichilnisky (ej@stanford.edu).

## Acknowledgements

This material adapts Jonathan Pillow's
[`GLMspiketraintutorial`](https://github.com/pillowlab/GLMspiketraintutorial)
(and its [Python port](https://github.com/pillowlab/GLMspiketraintutorial_python)).
Please also consider citing [`pynapple`](https://pynapple.org/citing.html) and
[`NeMoS`](https://nemos.readthedocs.io/en/latest/citation.html).

## License

[MIT](LICENSE)