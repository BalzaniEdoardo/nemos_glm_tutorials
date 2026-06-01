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

Below is a quick bit of data wrangling with `pynapple` that loads and temporally aligns the time series. The final result will be a [`TsGroup`](https://pynapple.org/generated/pynapple.TsGroup.html) that contains the spike times from 4 RGC units, the corresponding spike counts as a [`TsdFrame`](https://pynapple.org/generated/pynapple.TsFrame.html), and a [`Tsd`](https://pynapple.org/generated/pynapple.Tsd.html) with the stimulus. 

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

# Count the spikes
bin_size = stimulus.t[1] - stimulus.t[0]
counts = units.count(bin_size, stimulus.time_support)

# Re-sample the stimulus
stimulus = counts.value_from(stimulus, mode="before")

```

## Compute and plot the cross-correlogram

As a first step, let's take a look at the cross-correlograms (CCGs), and let's compute them via the `pynapple` functions [`compute_crosscorrelogram`](https://pynapple.org/generated/pynapple.process.correlograms.html#pynapple.process.correlograms.compute_crosscorrelogram) and [`compute_autocorrelogram`](https://pynapple.org/generated/pynapple.process.correlograms.html#pynapple.process.correlograms.compute_autocorrelogram).

```{code-cell} ipython3

# 30 bins matching the original tutorial
window_size_sec = 30 * bin_size

ccgs = nap.compute_crosscorrelogram(units, binsize=bin_size, windowsize=window_size_sec)
acgs = nap.compute_autocorrelogram(units, binsize=bin_size, windowsize=window_size_sec)

# drop acgs at t=0 
acgs.loc[0] = np.nan

ccgs
```

As you can see, the CCGs are stored in a pandas dataframe. Each column represents a pair of units, with the column name indicating the pair. Let's plot them.

```{code-cell} ipython3

fig = plt.figure(figsize=[12,8])
for i in acgs.columns:
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
```

Adding the spike history sharpens the prediction. Let's look at the predicted rate of both models against the observed spike counts. We reuse the `plot_counts` helper from the [first tutorial](tutorial-01), now shared via the `nemos_tutorials` package.

```{code-cell} ipython3
# Stimulus-only model is fit on the stim sub-block of the design matrix.
X_stim = bas.split_by_feature(X, axis=1)["stim"]

rate_stim_only = model_stim_only.predict(X_stim)
rate_stim_spk = model_stim_spk.predict(X)

# Pick a 1-second window starting after the NaN-padded history bins.
t0 = rate_stim_spk.dropna().t[0]
ep = (t0, t0 + 1)

plot_counts(
    neuron_counts,
    ep,
    [
        (rate_stim_only, "stim only"),
        (rate_stim_spk, "stim + spike hist"),
    ],
    title="single-neuron GLM: rate prediction",
    ylabel="spikes / bin",
)
plt.show()
```

# Fit coupled GLM for multiple-neuron responses

Instead of using the spike history of the fitted neuron only (auto-correlation filter), we will learn the functional connectivity by including the spike history of all the other neurons. In NeMoS this is trivial, since every basis is applied in a vectorized way over any extra axis:

- If `x` is 1D, then `basis.compute_features(x)` will return a $(\text{n_samples}, \text{n_basis_funcs})$ array.
- If `x` is ND with shape $(\text{n_samples}, i_1,...,i_{n-1})$, then the output will have shape $(\text{n_samples}, i_1 \cdot \dots \cdot i_{n-1} \cdot \text{n_basis_funcs})$.

Therefore, including all counts as predictors follows exactly the same syntax as the single count array case. The only caveat concerns the basis bookkeeping: we build the coupled design from a freshly constructed basis, `bas_coupling`, leaving the single-neuron `bas` untouched.

```{code-cell} ipython3

# Let's re-create the basis (see admonition below for why)
bas_coupling = bas_stim + bas_spk

X_coupling = bas_coupling.compute_features(stimulus, counts)
X_coupling
```

:::{admonition} Why are we re-defining the basis?
:class: note
:class: dropdown

Every call to `compute_features` inspects the shape of its inputs and stores it on the basis, so that a later `split_by_feature` knows how to carve the design matrix (or the coefficient vector) back into per-feature blocks of the right shape. This stored shape is *overwritten* on each call. If we reused the same `bas` object to build the coupled design, its `spike` block would be reshaped for 4 neurons, and we would no longer be able to split the *single-neuron* model's coefficients with it. Building the coupled design from a fresh `bas_coupling` keeps both models' bookkeeping intact.
:::

Now the number of columns is 105 = 25 + 20 * 4, where 4 is the number of units.

Again, we can split this design in interpretable components.

```{code-cell} ipython3

split_coupling = bas_coupling.split_by_feature(X_coupling, axis=1)
print("Keys:", split_coupling.keys())
```

But this time, the `spike` component of the design matrix is conveniently reshaped as `(n_samples, n_neurons, n_basis_funcs)`, one spike-history block per neuron.

```{code-cell} ipython3

print("(n_samples, n_neurons, n_basis_funcs): ", split_coupling["spike"].shape)
```

Again let's reverse the column order to match the original notebook. This is slightly more involved than in the single-neuron case, because now the two blocks have different rank: `stim` is a 2D array `(n_samples, n_basis)`, while `spike` is 3D `(n_samples, n_neurons, n_basis)`. In both we want to reverse the same axis, the last one (the basis axis), and `arr[..., ::-1]` does exactly that, whatever the array's shape.

On top of that, we want to rebuild a single 2D design matrix `(n_samples, n_regressors)`. Keeping the first (time) axis and flattening all the others together is precisely what `.reshape((n_samples, -1))` does. Putting the two steps together — reverse the last axis, then flatten — and stacking the blocks side by side gives the full coupled design.

```{code-cell} ipython3

n_samples = X_coupling.shape[0]
X_coupling = np.hstack(
    [Xi[..., ::-1].reshape((n_samples, -1)) for Xi in split_coupling.values()]
)
X_coupling
```

Let's plot it.

```{code-cell} ipython3

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
```

And, as before, let's compare the predicted rates of all three models on the same window we used above.

```{code-cell} ipython3
rate_coupled = model_coupled.predict(X_coupling)

plot_counts(
    neuron_counts,
    ep,
    [
        (rate_stim_only, "stim only"),
        (rate_stim_spk, "stim + spike hist"),
        (rate_coupled, "stim + coupling"),
    ],
    title="coupled GLM: rate prediction",
    ylabel="spikes / bin",
)
plt.show()
```

## Comparing the models

The filters and rate traces show that each added predictor changes the fit, but how much do we actually gain? Let's quantify it with two complementary metrics: the **single-spike information** (how many bits per spike the model buys us over a constant-rate baseline) and the **AIC** (which rewards fit but charges for extra parameters). Both build on the same quantity — the total log-likelihood of each fitted model — so let's compute that first.

One thing to be careful about: the convolution pads the start of each design with NaNs. Rather than hardcoding the window length, let's let the data tell us which bins are valid. Calling `dropna` on a design returns its non-NaN time support, and since all three designs share the same padding (the stimulus history is the longest window), any of them defines the common `valid_epochs`.

```{code-cell} ipython3
valid_epochs = X.dropna().time_support
```

Now we can restrict all the time series to `valid_epochs` and evaluate each model on the same bins. We keep the `(model, design)` pair for each fit in a dict, since we'll reuse both below.

```{code-cell} ipython3
counts_valid = neuron_counts.restrict(valid_epochs)

# (model, design matrix) for each of the three fits
fits = {
    "stim only": (model_stim_only, X_stim.restrict(valid_epochs)),
    "stim + spike hist": (model_stim_spk, X.restrict(valid_epochs)),
    "stim + coupling": (model_coupled, X_coupling.restrict(valid_epochs)),
}
```

By default, `score` returns the *mean* log-likelihood per sample. Here we want the total, so we pass `aggregate_sample_scores=np.sum` to sum the per-sample scores instead of averaging them.


```{code-cell} ipython3
log_likelihood = {
    name: model.score(design, counts_valid, aggregate_sample_scores=np.sum)
    for name, (model, design) in fits.items()
}
```

### Single-spike information

A raw log-likelihood is hard to read on its own, so we compare each model against a baseline that ignores everything and just fires at the constant mean rate. The difference between the two log-likelihoods, divided by the number of spikes and converted to base 2, is the **single-spike information**: the bits per spike we gain by knowing the model's rate rather than the mean rate ([Brenner et al., "Synergy in a Neural Code", Neural Comp 2000](https://www.princeton.edu/~wbialek/our_papers/brenner+al_00b.pdf)).

The baseline is a homogeneous Poisson model firing at the mean spike count. It is not a fitted GLM, so we evaluate its log-likelihood directly on the observation model.

```{code-cell} ipython3
n_samples = counts_valid.shape[0]
n_spikes = counts_valid.sum()

# Homogeneous Poisson baseline: constant rate = mean spike count.
mean_rate = np.mean(counts_valid) * np.ones(n_samples)
ll_null = model_stim_spk.observation_model.log_likelihood(
    counts_valid.d, 
    mean_rate,
    aggregate_sample_scores=np.sum
)

print("empirical single-spike information:\n-----------------------------------")
for name, ll in log_likelihood.items():
    ss_info = float((ll - ll_null) / n_spikes / np.log(2))
    print(f"{name:<18}: {ss_info:.2f} bits/sp")
```

Each added filter buys a bit more information per spike, with the biggest jump coming from the spike history.

### AIC

Single-spike information rewards a model for fitting the spikes but says nothing about how many parameters that fit cost. The Akaike Information Criterion charges for complexity,

$$
\text{AIC} = -2\,\log\text{-likelihood} + 2k,
$$

where $k$ is the number of free parameters and lower is better. We read $k$ straight off each fitted model: its filter weights plus the intercept.

```{code-cell} ipython3
def count_pars(model):
    """Number of free parameters: filter weights plus intercept."""
    return model.coef_.size + model.intercept_.size

aics = {
    name: float(-2 * log_likelihood[name] + 2 * count_pars(model))
    for name, (model, _) in fits.items()
}

for name, aic in aics.items():
    print(f"AIC  {name:<18}: {aic:.1f}")

winner = min(aics, key=aics.get)
print(f"\nAIC favors the '{winner}' model.")
```

Both metrics agree: the spike-history and coupling terms each add structure the stimulus alone cannot capture — the spike-history filter accounts for the cell's own refractoriness and bursting, while the coupling filters absorb shared variability from the rest of the population — and the improvement in fit more than pays for the extra parameters.

