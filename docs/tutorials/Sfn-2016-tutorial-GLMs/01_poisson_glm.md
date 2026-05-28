---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Tutorial 1 - Poisson GLM

This tutorial is an adaptation of [JW Pillow](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/tutorial1_PoissonGLM.ipynb)'s matherial presented at a short course on 

This is a tutorial illustrating the fitting of a linear-Gaussian GLM (also known as linear least-squares regression model) and a Poisson GLM (aka  "linear-nonlinear-Poisson" model) to retinal ganglion cell (RGC) spike trains stimulated with binary temporal white noise. 

(Data from [Uzzell & Chichilnisky, 2004](http://jn.physiology.org/content/92/2/780.long); see `README.txt` file in the `/data_RGCs` directory for details).
The dataset can be downloaded [here](https://pillowlab.princeton.edu/data/data_RGCs.zip):

The dataset is provided for tutorial purposes only, and should not be
distributed or used for publication without express permission from EJ
Chichilnisky (ej@stanford.edu).

## Downloading the dataset

The `nemos_tutorials` package ships a few utility functions that simplify dowloading the dataset and loading them into `pynapple`.  `pynapple` will be the entry point for most, if not all, the tutorials, and will take care of all the common pre-processing steps, such as counting, smoothing, up/down sampling etc.

Let's use the `fetch_data` utility to download the files and retrieve the download paths.

```{code-cell} ipython3
from nemos_tutorials import fetch_data

data_paths = fetch_data("data_RGCs")
data_paths
```


## Loading data into pynapple

In this first tutorial, we will demonstrate how to load the RGC data into [`pynapple`](https://pynapple.org) directly from the original Matlab files via `scipy.io.loadmat`. Subsequently, we will use directly an utility function for brevity. 


```{code-cell} ipython3
import pynapple as nap
from scipy.io import loadmat


```