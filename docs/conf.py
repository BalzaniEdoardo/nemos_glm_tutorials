# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "NeMoS GLM Tutorials"
copyright = "2026, Edoardo Balzani"
author = "Edoardo Balzani"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",                  # parse + execute MyST-Markdown / notebooks
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_togglebutton",      # collapsible admonitions (`:class: dropdown`)
]

# Treat .md files as MyST notebooks (executed via myst-nb).
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/*_files"]

# -- MyST / MyST-NB configuration --------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# Execute notebooks at build time. Use "cache" to re-run only when changed,
# "off" to render the stored outputs without executing.
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "nemos": ("https://nemos.readthedocs.io/en/latest/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "NeMoS GLM Tutorials"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/BalzaniEdoardo/nemos_glm_tutorials",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "path_to_docs": "docs",
}