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

## Load and pre-process the data

Below is a quick bit of data wrangling with `pynapple` that loads and temporally aligns the time series. The final result will be a [`TsGroup`](https://pynapple.org/generated/pynapple.TsGroup.html) that contains the spike times from 4 RGC units, the corresponding spike counts as a [`TsdFrame`](https://pynapple.org/generated/pynapple.TsdFrame.html), and a [`Tsd`](https://pynapple.org/generated/pynapple.Tsd.html) with the stimulus. 

For more details on the `pynapple` objects and a step-by-step walkthrough of the pre-processing, see the [first tutorial](tutorial-01).

```{code-cell} ipython3

import matplotlib.pyplot as plt
import numpy as np
from nemos_tutorials import fetch_data, PALETTE, plot_counts
import pynapple as nap
import jax
from scipy.io import loadmat

# enable float64 for precision
jax.config.update("jax_enable_x64", True)

data_paths = fetch_data("data_RGCs")

# Load spike times
spike_times = loadmat(data_paths["SpTimes.mat"], simplify_cells=True)["SpTimes"]
units = nap.TsGroup({i: nap.Ts(val) for i, val in enumerate(spike_times)})

# Load stimulus times and values
stim_times = loadmat(data_paths["stimtimes.mat"], simplify_cells=True)["stimtimes"]
stim = loadmat(data_paths["Stim.mat"], simplify_cells=True)["Stim"]
stimulus = nap.Tsd(stim_times, stim)

# Align time support
units = units.restrict(stimulus.time_support)

```

## Upsample to get finer timescale representation of stim and spikes


The need to regularize GLM parameter estimates is acute when we don't have enough data relative to the number of parameters we're trying to estimate, or when using correlated (eg naturalistic) stimuli, since the stimuli don't have enough power at all frequencies to estimate all frequency components of the filter. 

The RGC dataset we've looked at so far requires only a temporal filter (as opposed to spatio-temporal filter for full spatiotemporal movie stimuli), so it doesn't have that many parameters to estimate. It also has binary white noise stimuli, which have equal energy at all frequencies.

Regularization thus isn't an especially big deal for this data (which was part of our reason for selecting it). However, we can make it look correlated by considering it on a finer timescale than the frame rate of the monitor.  (Indeed, this will make it look highly correlated).


Let's first restrict all time series to the first minute, this will allow for a faster runtime for the notebook, then let
s upsample.

```{code-cell} ipython3
# Create a 1min long interval set
epoch_1min = nap.IntervalSet(stimulus.t[0], stimulus.t[0] + 60)

# Restrict the time series
units = units.restrict(epoch_1min)
stimulus = stimulus.restrict(epoch_1min)

# Conut with 10x resolution
bin_size = (stimulus.t[1] - stimulus.t[0]) / 10

# Count the spikes
counts = units.count(bin_size, stimulus.time_support)

# Re-sample the stimulus
stimulus = counts.value_from(stimulus, mode="before")
```
:::{admonition} Pre-processing comparison
:class: tip

Note how the pre-processing pipeline for the upsampled case looks almost identical to the default case by comparing this step with [second tutorial](tutorial-02). Once the spikes are counted at the right resolution, the re-sampling of other time series is derived from it. No need for special interpolation calls.

:::

And now let's visualize the upsampled data.

```{code-cell} ipython3
# Visualize the upsampled data.

fig = plt.figure(figsize=[12,8])
plt.subplot(211)
# plot 0.5 sec
plt.plot(stimulus.get(0, 0.5), linewidth=4)
plt.title('raw stimulus (fine time bins)')
plt.ylabel('stim intensity')
plt.subplot(212)
plt.stem(counts.get(0,0.5).t, counts[:, 2].get(0,0.5).d)
plt.title('binned spike counts')
plt.ylabel('spike count')
plt.xlabel('time (s)')
plt.tight_layout()
plt.show()
```

Let's divide in train and test set. This is the simplest way of cross-validating, later we will see improved cross-validation schemes.

```{code-cell} ipython3
# Get the total duration (60 sec)
ep_tot = stimulus.time_support
train_frac = 0.8

train_ep = nap.IntervalSet(
    ep_tot.start, 
    ep_tot.start + ep_tot.tot_length() * train_frac
)
# perform a set difference to get the test set
test_ep = ep_tot.set_diff(train_ep)

print("Train:\n", train_ep)
print("\n\nTest:\n", test_ep)
```