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

# Poisson GLM with Spike History and Coupling

This tutorial is an adaptation of [JW Pillow](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/tutorial2_spikehistcoupledGLM.ipynb)'s material, presented at the *Data Science and Data Skills for Neuroscientists* short course at the SfN 2016 meeting.


 This is an interactive tutorial designed to walk you through the steps of fitting an autoregressive Poisson GLM (i.e., a spiking GLM with spike-history) and a multivariate autoregressive Poisson GLM (i.e., a GLM with spike-history AND coupling between neurons).

 (Data from [Uzzell & Chichilnisky, 2004](http://jn.physiology.org/content/92/2/780.long); see `README.txt` file in the `/data_RGCs` directory for details).
The dataset can be downloaded [here](https://pillowlab.princeton.edu/data/data_RGCs.zip):

The dataset is provided for tutorial purposes only, and should not be distributed or used for publication without express permission from EJ Chichilnisky (ej@stanford.edu).

## Load and pre-process the data

Below a quick data wrangling with `pynapple` that loads, and temporally align the time series. The final result will be a [`TsGroup`](https://pynapple.org/generated/pynapple.TsGroup.html) that contains the spike times from 4 RGCs units, the corresponding spike counts as a [`TsdFrame`](https://pynapple.org/generated/pynapple.TsFrame.html) and a [`Tsd`](https://pynapple.org/generated/pynapple.Tsd.html) with the stimulus. 

For more details on the `pynapple` objects and a step-by-step walkthrough of the pre-processing, see the [first tutorial](tutorial-01).

```{code-cell} ipython3

import maplotlib.pyplot as plt
from nemos_tutorials import fetch_data
import pynapple as nap
import jax
from scipy.io import loadmat

PALETTE = plt.cm.Pastel1.colors

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

# Count the spikes
bin_size = stimulus.t[1] - stimulus.t[0]
counts = units.count(bin_size, stimulus.time_support)

# Re-sample the stimulus
stimulus = counts.value_from(stimulus, mode="before")

```

## Compute and plot the cross-correlogram

As a first step, let's take a look at the cross-correlograms (CCGs), and let's compute them via the `pynappple` functions [`compute_crosscorrelogram`](https://pynapple.org/generated/pynapple.process.correlograms.html#pynapple.process.correlograms.compute_crosscorrelogram) and [`compute_autocorrelogram`](https://pynapple.org/generated/pynapple.process.correlograms.html#pynapple.process.correlograms.compute_autocorrelogram)..

```{code-cell} ipython3

# 30 bins matching the original tutorial
window_size_sec = 30 * bin_size

ccgs = nap.compute_crosscorrelogram(units, binsize=bin_size, windowsize=window_size_sec)
acgs = nap.compute_autocorrelogram(units, binsize=bin_size, windowsize=window_size_sec)

# drop acgs at t=0 
acgs.loc[0] = np.nan

ccgs
```

As you can see, the CCGs are stored in a pandas dataframe. Each column represent a pair of units, with the column name indicate the paris. Let's plot them.

```{code-cell} ipython3

import matplotlib.pyplot as plt

fig = plt.figure(figsize=[12,8])
for i in acg.columns:
    plt.subplot(len(units), len(units), i*len(units) + i + 1)
    plt.title(f'cells ({i},{i})')
    plt.plot(acgs[i])
    plt.xlabel('time shift (s)')
    
for i, j in ccgs.columns:
    plt.subplot(len(units), len(units), i*len(units) + j + 1)
    plt.title(f'cells ({i},{j})')
    plt.plot(ccgs[i, j])
    plt.xlabel('time shift (s)')
plt.tight_layout()
plt.show()

```

## Building the design matrix

Let's build a design matrix that uses as predictors the history of the stimulus (as in the first tutorial), and the spike count history of the neuron we are fitting. The last term will capture the autocorrelation structure of the spike count time series.

As before, we can use the NeMoS [`HistoryConv`](https://nemos.readthedocs.io/en/latest/generated/basis/nemos.basis.HistoryConv.html) basis to capture the history effect of both predictors. One way to construct this is to create two basis, create the design matrices, and then concatenate them. An alternative and easier way to construct the same predictor in NeMoS is to use basis addition: adding two basis together result in a composite basis that concatenate design matrices.

```{code-cell} ipython3
import nemos as nmo

# match the original tutorial
cell_idx = 2
neuron_counts = counts[:, cell_idx]

# number of history time bins used for prediction
window_size_stim = 25 
window_size_spk = 20 

# Define the basis and specify labels to help tracking
# the component identity

# Basis for the stimulus
# (do not shift by 1 after convolve: stim_t is used as predictor of counts_t)
bas_stim = nmo.basis.HistoryConv(window_size_stim, conv_kwargs={"shift": False}, label="stim")

# Basis for the spike hist 
# (shift after convolve: count_t is NOT used to predict count_t)
bas_spk = nmo.basis.HistoryConv(window_size_spk, label="spike")

# add
bas = bas_stim + bas_spk
bas
```

Once we have our additive basis, we can call compute feature passing the predictors in the correct order to get the full design matrix.

```{code-cell} ipython3

X = bas.compute_features(stimulus, neuron_counts)
X
```

The total number of columns is 45, 25 columns capturing the stimulus history, and 20 for the spike history. The bookkeeping of which columns belong to which predictor may become messy, and bug-prone, especially for multiple high-dimensional predictors. Fortunately, the basis object takes care of the bookkeeping for us. Let's see how to use the `split_by_feature` method to partition the design matrix into the two components and let's plot them.

```{code-cell} ipython3
# Split by feature and return a dict with keys the basis label
split_dict = bas.split_by_feature(X, axis=1)

# Loop over the two predictors and print their shapes
for key, Xi in split_dict.items():
    print(key, Xi.shape)

```

Finally, let's revert the column order for each predictor, as we did in the [first tutorial](design-matrix-tutorial-01). As before, this doesn't change the model, but matches the design construction of the original implementation, which is handy if one wants to compare the two notebooks. 

```{code-cell} ipython3
# Revert the column order and concatenate
X = np.hstack([Xi[...,::-1] for Xi in split_dict.values()])
```

Finally, let's plot the design. Let's remember that NeMoS performs a convolution in mode `valid`, and append NaNs to preserve the total number of samples, which naturally maintains the temporal alignment with the predicted variable.

```{code-cell} ipython3
# skip the first NaNs
X_slice = X[window_size_stim:window_size_stim+50]
counts_slice = neuron_counts[window_size_stim:window_size_stim+50]

fig = plt.figure(figsize=[12,8])
plt.subplot(1, 10, (1,9))
vmin = min(X_slice.min(), counts_slice.min())
vmax = max(X_slice.max(), counts_slice.max())
plt.imshow(
    X_slice, 
    aspect='auto', 
    interpolation='nearest', 
    vmin=vmin,
    vmax=vmax,
)
plt.xlabel('regressor')
plt.ylabel('time bin of response')
plt.title('design matrix (including stim and spike history)')
plt.subplot(1,10,10)
plt.imshow(
    counts_slice[:, None], 
    aspect='auto', 
    interpolation='nearest',
    vmin=vmin,
    vmax=vmax,
)
plt.yticks(ticks=[], labels=[])
plt.title('spike count')
plt.tight_layout()
plt.show()
```

## Fit a single-neuron GLM with spike-history

Now we are ready to fit our Poisson GLM. Let's fit two models: a model that uses the stimulus as predictor, and one that uses both the stimulus and the spike history.

We can take advantage of the basis bookkeeping again to split the design matrix.
```{code-cell} ipython3

model_stim_only = nmo.glm.GLM(solver_name="BFGS")
model_stim_spk =  nmo.glm.GLM(solver_name="BFGS")

model_stim_only.fit(bas.split_by_feature(X, axis=1)["stim"], neuron_counts)
model_stim_spk.fit(X, neuron_counts)
```

And finally, let's plot and compare the filters.

```{code-cell} ipython3

# split coefficients for composite model
coef_dict = bas.split_by_feature(model_stim_spk.coef_, axis=0)

f, (ax1, ax2) = plt.subplots(2,1)
lags_stim = np.arange(-1 * window_size_stim + 1,1) * bin_size
lags_spk = np.arange(-1 * window_size_spk + 1,1) * bin_size
ax1.plot(lags_stim, model_stim_only.coef_, marker="o", color=PALETTE[0], label='stim only')
ax1.plot(lags_stim, coef_dict["stim"], marker="o", color=PALETTE[1], label='stim + sp hist')
ax1.legend(loc='upper left')
ax1.set_title('stimulus filters')
ax1.set_ylabel('weight')
ax1.set_xlabel('time before spike (s)')
ax2.plot(lags_spk, coef_dict["spike"], marker="o", color=PALETTE[1])
ax2.set_title('spike history filter')
ax1.set_xlabel('time before spike (s)')
ax1.set_ylabel('weight')
plt.tight_layout()
plt.show()

## TODO FOR CLAUDE: add the plotting of the rate + count 
## INSTRUCTUION: Move the plotting function from tutorial 01 to src/nemos_tutorials/plotting.py 
## creating the script. Use that function here too.
```

# Fit coupled GLM for multiple-neuron responses

Instead of using the spike history of the fitted neuron only (auto-correlation filter), we will learn the functional connectivity by including the spike history of all the other neurons. In nemos this is trivial, since every basis is applied in a vectorized way over any extra axis:

- If `x` is 1D, then `basis.compute_features(x)` will return a $(\text{n_samples}, \text{n_basis_funcs})$ array.
- If `x` is ND with shape $(\text{n_samples}, i_1,...,i_{n-1})$, then the output will have shape $(\text{n_samples}, i_1 \cdot \dots \cdot i_{n-1} \cdot \text{n_basis_funcs})$.

Therefore, including all counts as predictors follows exactly the same syntax as the single count array case. The only caveat is that we need to re-create the basis otherwise the bookkeeping of the original basis that keeps track of the coefficient structure will be overridden. 

TODO CLAUDE: explain this better (every call of compute features extract teh input shape and uses it for bookkeeping, therefore if we re-use the `bas`, we won't be able to split the original model coefficients.)

```{code-cell} ipython3

# Let's re-create the basis 
bas_coupling = bas_stim + bas_spk

X_coupling = bas_coupling.compute_features(stimulus, counts)
X_coupling
```

Now the number of columns is 105 = 25 + 20 * 4, where 4 is the number of units.

Again, we can split this design in interpretable components.

```{code-cell} ipython3

split_coupling = bas_coupling.split_by_feature(X_coupling, axis=1)
print("Keys:", split_coupling.keys())
```

But this time, the `stim` component of teh design matrix is conveniently reshaped as `(n_samples,n_neurons, n_basis_funcs)`.

```{code-cell} ipython3

print("(n_samples, n_neurons, n_basis_funcs): ", split_coupling["spike"].shape)
```

Again let's reverse the column order to match the original notebook, and plot the model design.

```{code-cell} ipython3

n_samples = X_coupling.shape[0]
X_coupling = np.hstack(
    [Xi[...,::-1].reshape((n_samples, -1)) for Xi in split_coupling.values()]
)

# Take first 50 valid samples
X_slice = X_coupling[window_size_stim: window_size_stim+50]
fig = plt.figure(figsize=[12,8])
plt.imshow(X_slice, aspect='auto', interpolation='nearest')
plt.xlabel('regressor')
plt.ylabel('time bin of response')
plt.title('design matrix (stim and 4 neurons spike history)')
plt.show()
```

Finally, let's fit the coupled model.

```{code-cell} ipython3

model_coupled =  nmo.glm.GLM(solver_name="BFGS")
model_coupled.fit(X_coupling, neuron_counts)
```


And let's compare all the results.

```{code-cell} ipython3

coef_coupling_dict = bas_coupling.split_by_feature(model_coupled.coef_, axis=0)

f, (ax1, ax2) = plt.subplots(2,1)
lags_stim = np.arange(-1 * window_size_stim + 1,1) * bin_size
lags_spk = np.arange(-1 * window_size_spk + 1,1) * bin_size
ax1.plot(lags_stim, model_stim_only.coef_, marker="o", color=PALETTE[0], label='stim only')
ax1.plot(lags_stim, coef_dict["stim"], marker="o", color=PALETTE[1], label='stim + sp hist')
ax1.plot(lags_stim, coef_coupling_dict["stim"], marker="o", color=PALETTE[2], label='stim + coupling')
ax1.legend(loc='upper left')
ax1.set_title('stimulus filters')
ax1.set_ylabel('weight')
ax1.set_xlabel('time before spike (s)')
ax2.plot(lags_spk, coef_dict["spike"], marker="o", color=PALETTE[1])
for i, coef in enumerate(coef_coupling_dict["spike"]):
    ax2.plot(lags_spk, coef, marker="o", ls="--", color=PALETTE[2+i])
ax2.set_title('spike history filter')
ax1.set_xlabel('time before spike (s)')
ax1.set_ylabel('weight')
plt.tight_layout()

plt.show()



## TODO FOR CLAUDE: add the plotting of the rate + count 
## INSTRUCTUION: Move the plotting function from tutorial 01 to src/nemos_tutorials/plotting.py 
## creating the script. Use that function here too.

```


# TODO FOR CLAUDE: add the AIC session.

