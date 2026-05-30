# GLM Tutorials

The material proposed in these tutorials revisits what was presented by [Jonathan Pillow](https://pillowlab.princeton.edu) in a "short course" on [Data Science and Data Skills for Neuroscientists](https://neuronline.sfn.org/scientific-research/data-science-and-data-skills-for-neuroscientists) organized at the SFN 2016 meeting, constructing and fitting models with [`pynapple`](https://pynapple.org/) and [`NeMoS`](https://nemos.readthedocs.io/en/latest/index.html).

The original Matlab implementation and its python translation can be found at the following links:

- Matlab: https://github.com/pillowlab/GLMspiketraintutorial
- Python:  https://github.com/pillowlab/GLMspiketraintutorial_python


<h2 style="font-size: 2em; font-weight: bold; margin-top: 20px; margin-bottom: 10px;">Contents</h2>


```{toctree}
:maxdepth: 2

01_poisson_glm.md
```

## What's changed

- As of NeMoS version `0.2.7`, the smoothing Laplacian prior is not shipped with the package. This prior was used in the original `tutorial3_regularization_linGauss` and `tutorial4_regularization_PoissonGLM`. Smoothing is obtained by basis expansion approximation instead. 

## Citation

If you found this material useful and you wish to cite this tutorial, feel free to :
- Acknowledge the paper from which it was developed: [Pillow et al, *Nature* 2008](https://pillowlab.princeton.edu/pubs/abs_Pillow08_nature.html)
- Acknowlege `pynapple` package by citing the [accompanying paper](https://pynapple.org/citing.html). 
- Acknowledge `NeMoS` package by citing the [associated DOI](https://nemos.readthedocs.io/en/latest/citation.html).