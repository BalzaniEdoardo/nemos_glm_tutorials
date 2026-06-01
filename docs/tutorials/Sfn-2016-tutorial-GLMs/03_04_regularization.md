---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  language: python
  name: python3
  display_name: Python 3 (ipykernel)
---

# Tutorial 3+4 - Gaussian and Poisson GLM with Regularization


This tutorial is adapts and combines two notebooks from JW Pillow's material, presented at the *Data Science and Data Skills for Neuroscientists* short course at the SfN 2016 meeting:

- [tutorial3_regularization_linGauss.ipynb](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/tutorial3_regularization_linGauss.ipynb).
- [tutorial4_regularization_PoissonGLM.ipynb](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/tutorial4_regularization_PoissonGLM.ipynb).


 This is an interactive tutorial designed to walk you through how to fit a GLM controlling for under/overfitting via regularization and cross-validation. In particular, we will illustrate two forms of regularization: ridge and laplacian smoothing.

 (Data from [Uzzell & Chichilnisky, 2004](https://pubmed.ncbi.nlm.nih.gov/15277596/); see `README.txt` file in the `/data_RGCs` directory for details).
The dataset can be downloaded [here](https://pillowlab.princeton.edu/data/data_RGCs.zip):

The dataset is provided for tutorial purposes only, and should not be distributed or used for publication without express permission from EJ Chichilnisky (ej@stanford.edu).
