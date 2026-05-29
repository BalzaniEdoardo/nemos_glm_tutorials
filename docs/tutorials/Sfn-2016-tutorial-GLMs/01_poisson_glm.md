---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---

# Tutorial 1 - Poisson GLM

This tutorial is an adaptation of [JW Pillow](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/tutorial1_PoissonGLM.ipynb)'s material, presented at the *Data Science and Data Skills for Neuroscientists* short course at the SfN 2016 meeting.

It illustrates how to fit a linear-Gaussian GLM (also known as a linear least-squares regression model) and a Poisson GLM (also known as a "linear-nonlinear-Poisson" model) to retinal ganglion cell (RGC) spike trains driven by binary temporal white noise.

(Data from [Uzzell & Chichilnisky, 2004](http://jn.physiology.org/content/92/2/780.long); see [`README.txt`](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/data_RGCs/README.txt) for details).

The dataset is provided for tutorial purposes only, and should not be distributed or used for publication without express permission from EJ Chichilnisky (ej@stanford.edu).

## Downloading the dataset

The `nemos_tutorials` package ships a few utility functions that simplify downloading the dataset and loading it into `pynapple`. `pynapple` will be the entry point for most, if not all, of these tutorials, taking care of common pre-processing steps such as counting, smoothing, and up/down-sampling.

Let's use the `fetch_data` utility to download the files and retrieve their local paths.

```{code-cell} ipython3
import jax
from nemos_tutorials import fetch_data

# enable float64 for precision
jax.config.update("jax_enable_x64", True)

data_paths = fetch_data("data_RGCs")
data_paths
```

## Loading data into pynapple

In this first tutorial, we will load the RGC data into [`pynapple`](https://pynapple.org) directly from the original Matlab files via `scipy.io.loadmat`, so that every step is explicit. In later tutorials we will rely on a utility function instead, for brevity.


### Spike times as a TsGroup

Let's start with the spike times. These load as a list of 4 arrays, one spike train per recorded unit.

```{code-cell} ipython3
from scipy.io import loadmat

# Load the array of spike times (one array per unit, 4 units total)
spike_times = loadmat(data_paths["SpTimes.mat"], simplify_cells=True)["SpTimes"]

print("Number units: ", len(spike_times))

# Print the spike times of the first unit.
print("Spike times of unit 0:")
print(spike_times[0])
```

A list of spike times can be loaded directly into a `pynapple` [TsGroup](https://pynapple.org/generated/pynapple.TsGroup.html), a dictionary-like object that holds multiple timestamp arrays together with their associated metadata, if any.

```{code-cell} ipython3
import pynapple as nap

# Load spike times into pynapple
units = nap.TsGroup({i: nap.Ts(val) for i, val in enumerate(spike_times)})
units
```

The [TsGroup](https://pynapple.org/generated/pynapple.TsGroup.html) is displayed as a table. The first column is the unit index; if not provided, `pynapple` assigns `0, ..., num_units - 1`. The second column is each unit's mean firing rate in Hz, computed directly from its spike train. Any additional columns hold the associated metadata.


### Stimulus as a Tsd

For this dataset, the stimulus is a full-field binary white noise. The stimulus presentation times and the stimulus values are stored in two separate 1-D arrays.

```{code-cell} ipython3
# Load stimulus times and values
stim_times = loadmat(data_paths["stimtimes.mat"], simplify_cells=True)["stimtimes"]
stim = loadmat(data_paths["Stim.mat"], simplify_cells=True)["Stim"]

print(f"\ntimes: {stim_times[:5]}\n\nvalues: {stim[:5]}")
```

`pynapple` stores 1-D time series as `Tsd` objects. Each `Tsd` has a time attribute `t` holding the timestamps and a data attribute `d` holding the corresponding values.

```{code-cell} ipython3
stimulus = nap.Tsd(stim_times, stim)

print("Number of stim frames:", len(stimulus))
# Tsd.rate stores the sampling rate in Hz
# Note: that this is not computed directly as the delta between consecutive bins, 
# but as the total duration divided by the number of events.
print(f"Time bin size: {1000./stimulus.rate :.2} ms")
```

Finally, let's plot one second of the time series. We can use the `get` method to extract a specific time interval.

```{code-cell} ipython
import matplotlib.pyplot as plt

cell_idx = 2
plt.figure()

# Note: matplotlib conveniently uses stimulus.t (the index) as the x-axis.
plt.plot(stimulus.get(0, 1), label="stimulus")

# Overlay the spike raster at y=-0.6
plt.plot(units[cell_idx].get(0, 1).fillna(-0.6), "|", color="k", label="spikes")

plt.title("raw stimulus (full field flicker)")
plt.ylim(-0.66, 0.8)
plt.xlabel("time (s)")
plt.legend()
plt.show()

```
## Pre-processing

A Poisson GLM predicts a spike count at each time bin from the stimulus in that bin (and, later, from the recent stimulus and spike history). For the model inputs and outputs to line up, we need three things:

1. **Spike counts.** The spike times must be converted to spike counts, since counts are what the Poisson GLM models.
2. **Matched sampling.** The stimulus must be re-sampled onto the same time bins as the counts, so that each count has exactly one stimulus value associated with it.
3. **Temporal alignment.** The time axes of the counts and the re-sampled stimulus must cover the same interval, so that bin `i` of one corresponds to bin `i` of the other.

Let's start with alignment. Two time series are temporally aligned when they span the same interval. In `pynapple`, the interval covered by a time series is stored in its `time_support` attribute, a [`IntervalSet`](https://pynapple.org/generated/pynapple.IntervalSet.html) object: a collection of start/end pairs marking continuous recording epochs.

Since we did not pass a time support when constructing the time series, `pynapple` set each one to a single epoch spanning its full range. Let's print and compare the supports of our two time series:

```{code-cell} ipython3
print("Stimulus\n========")
print("Time support:\n", stimulus.time_support)
print(
    "\nTime stamps range:\n"
    f"({float(stimulus.t[0]):.3f}, {float(stimulus.t[-1]):.3f})"
)


unit_ts_range = min(spks.t[0] for spks in units.values()), max(spks.t[-1] for spks in units.values())

print("\nUnits\n=====")
print("Time support:\n", units.time_support)
print(
    "\nSpike times range:\n"
    f"({float(unit_ts_range[0]):.3f}, {unit_ts_range[1]:.3f})"
)
```

As we can see, the supports do not match: the spikes extend beyond the window in which the stimulus was actually presented. Before doing anything else, we align the two by restricting the `units` time series to the stimulus support. That is exactly what the `restrict` method is for.

```{code-cell} ipython3
units = units.restrict(stimulus.time_support)
units.time_support
```

Two things happened here: every spike that fell outside the stimulus support was dropped, and the time support of `units` was replaced by that of `stimulus`. The two series now span the same interval.

With the supports aligned, we can convert the spike times to counts using the `count` method of `TsGroup`. Given a bin size, `count` tiles the `time_support` with uniform bins and counts the spikes falling in each one. We choose a bin size equal to the stimulus frame interval, so that the counts will share the stimulus's native resolution.

```{code-cell} ipython3
bin_size = stimulus.t[1] - stimulus.t[0]
counts = units.count(bin_size, stimulus.time_support)
counts
```

The counts are now regularly binned over the whole time support. Let's plot the counts overlaid with the spike times.

```{code-cell} ipython3
plt.figure()
plt.plot(counts[:, cell_idx].get(0, 1), label="spike counts")
plt.plot(units[cell_idx].get(0, 1).fillna(-0.2), "|", color="k", label="spikes")
plt.title('binned spike counts')
plt.ylabel('spike count')
plt.xlabel('time (s)')
plt.legend()
plt.show()
```
The last step is to put the stimulus on these exact bins. We do this with `value_from`, which, for every timestamp in `counts.t`, looks up a value from the stimulus. Using `mode="before"` picks the most recent stimulus sample at or before each count bin, which is the causally correct choice: the count in a bin can only be driven by stimulus that has already been presented, never by a future frame.

```{code-cell} ipython3
# for every sample i, take the most recent stimulus value at or before counts.t[i]
stimulus = counts.value_from(stimulus, mode="before")
stimulus
```

And that's it: `counts` and `stimulus` are now aligned to the same interval, sampled on the same bins, and ready for GLM modeling.

## Building the design matrix

Now it's time to build the design matrix for our model. The predictor we want is the recent stimulus history over a fixed window of $w$ samples: to predict the spike count $y_t$ at time $t$, we use the stimulus values $s_{t-1},\dots, s_{t-w}$ that precede it.

The resulting design matrix looks like this,

$$
X = \begin{bmatrix}
s_{0} & s_1 & \dots & s_{w} \\
s_{1} & s_2 & \dots & s_{w+1} \\
\dots & \dots & \dots & \dots \\
s_{T-w} & s_{T-w+1} & \dots & s_{T} \\
\end{bmatrix} \label{eq:design-matrix}
$$

Two things to notice: 1) $X$ has $T-w$ rows, where $T$ is `len(stimulus)`, because we need at least $w$ stimulus values to fill a row. 2) Each row is a shifted copy of the row above.

A convenient way to construct this design matrix is to convolve the stimulus with an identity matrix and then reverse the column order. In `NeMoS`, convolution with the identity is exactly what the `HistoryConv` basis does.


```{code-cell} ipython3
import nemos as nmo

# Match the original notebook
window_size = 25

# define the basis object, shift = False means that
bas = nmo.basis.HistoryConv(window_size, conv_kwargs={"shift": False})

# Convolve with the identity
X = bas.compute_features(stimulus)
# Reverse column order (to match original notebook)
X = X[:,::-1]
```

As you can see:

1. The design matrix is still a `pynapple` object: the time-series information is preserved, including the time axis and the time support.
2. `X` has the same number of samples as `stimulus`. The convolution itself runs in `valid` mode, which produces only $T - w$ values, but `NeMoS` pads the result with NaNs back to length $T$. This keeps `X` and `counts` aligned.

Reversing the column order (or not) yields an equivalent design; it only changes how the columns are interpreted. Following $\eqref{eq:design-matrix}$, the first column holds the stimulus $w$ samples in the past, and the last column holds the most recent stimulus sample.

```{code-cell} ipython3

plt.pcolormesh(X[window_size:window_size+50], cmap="Pastel1")
plt.show()
```

## Compute and visualize the spike-triggered average (STA)

When the stimulus is white noise, the STA is an unbiased estimator of the filter in a GLM / LNP model, as long as the nonlinearity yields an STA whose expectation is nonzero. (Feel free to skip this technical aside: it simply means that a symmetric nonlinearity, e.g. $x^2$, breaks the condition and the STA becomes uninformative.)

Even when your stimulus is not white noise, it is often worth visualizing the STA: if it shows no structure at all, that is a sign something has gone wrong upstream, for example a mismatch between the design matrix and the binned spike counts.


```{code-cell} ipython3
import numpy as np

neuron_counts = counts[:, cell_idx]

# Skip the nans. Note that transposition in pynapple doesn't work, so we grabbed
# the data attribute `d` (which is a numpy array).
sta = (X.d[window_size:].T @ neuron_counts[window_size:]) / neuron_counts.sum()

ttk = np.arange(-window_size+1,1) / neuron_counts.rate  # time bins for STA (in seconds)

plt.figure()
plt.plot(ttk,ttk*0, 'k--')
plt.plot(ttk, sta, 'bo-')
plt.title('STA')
plt.xlabel('time before spike (s)')
plt.xlim([ttk[0],ttk[-1]])
plt.show()
```

:::{admonition} Why is our STA shifted by one bin from the original tutorial?
:class: note dropdown

If you put this STA side by side with the one in [Pillow's original notebook](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/tutorial1_PoissonGLM.ipynb), you'll notice ours is shifted by one bin: the peak sits one sample closer to zero lag. The two are otherwise identical, and reconciling them is instructive — the shift is a small, concrete example of a general issue: *how you align the spike counts to the stimulus in time directly shapes what you read off the result.*

The shift happens because the two pipelines anchor the spike histogram differently. In this dataset the stimulus frames are timestamped at `dt, 2·dt, 3·dt, …` (`stim_times[0] == dt`). The original notebook bins spikes on a grid anchored at zero (`np.arange(num_time_bins+1) * dt`), so its bins are offset by one frame from where the stimulus actually starts. Here we let `count` anchor the bins at the stimulus support (which starts at `stim.t[0] == dt`) and use `value_from(..., mode="before")` to pick the most recent frame at or before each bin, so the counts and the stimulus stay genuinely aligned in time.

At this bin size (~8 ms) the one-sample difference is negligible and doesn't change the shape of the filter or any conclusion we draw from it. But the size of the error is one bin *whatever the bin size is*: with coarser bins, the same misalignment would move the estimated STA peak by that much more, and could meaningfully distort the timing you report. This is exactly why `pynapple` is so handy — it tracks the real timestamps and gives you fine control over binning and resampling, so alignment is something you set deliberately rather than get by accident.
:::

## Whitened STA

If the stimuli are non-white, then the STA is generally a biased estimator for the linear filter. In this case we may wish to compute the "whitened" STA, which is also the maximum-likelihood estimator for the filter of a GLM with "identity" nonlinearity and Gaussian noise (also known as least-squares regression or linear regression).

If the stimuli have correlations this ML estimate may look like garbage (more on this later when we come to "regularization").  But for this dataset the stimuli are white, so we don't (in general) expect a big difference from the STA (this is because `X.T @ X` is  close to a scaled version of the identity).

```{code-cell} ipython3

# Whitened STA (or linear regression via the analytical formula)
wsta = np.linalg.pinv(X.d[window_size:].T @ X[window_size:]) @ sta * neuron_counts.sum()

plt.figure()
plt.plot(ttk,ttk*0, 'k--')
plt.plot(ttk, sta/np.linalg.norm(sta), 'bo-', label="STA")
plt.plot(ttk, wsta/np.linalg.norm(wsta), 'ro-', label="wSTA")
plt.title('STA and whitened STA')
plt.xlabel('time before spike (s)')
plt.xlim([ttk[0],ttk[-1]])
plt.legend()
plt.show()
```

## Rate prediction with a linear-Gaussian GLM

The whitened STA can actually be used to predict spikes because it corresponds to a proper estimate of the model parameters (i.e., for a Gaussian GLM). Let's inspect this prediction.


```{code-cell} ipython3

# Predicted spikes from linear-Gaussian GLM
sppred_lgGLM = X @ wsta  

# Drop initial nans
sppred_lgGLM = sppred_lgGLM

# get the first 1sec of non-nans
first_valid_time = sppred_lgGLM.dropna().t[0]
ep_1sec = first_valid_time, first_valid_time + 1

plt.figure()
markerline,_,_ = plt.stem(neuron_counts.get(*ep_1sec).t, neuron_counts.get(*ep_1sec), linefmt='b-', basefmt='k-', label="spike ct")
plt.setp(markerline, 'markerfacecolor', 'none')
plt.setp(markerline, 'markeredgecolor', 'blue')
plt.plot(sppred_lgGLM.get(*ep_1sec), color='red', linewidth=2, label="lgGLM")
plt.title('linear-Gaussian GLM: spike count prediction')
plt.ylabel('spike count'); plt.xlabel('time (s)')
plt.ylim(-1.2, 4)
plt.legend()
plt.show()
```

We can clearly see that we forgot to include an offset or "intercept" term to our design matrix, which will allow our prediction to have a non-zero mean (since the stimulus here was normalized to have zero mean).


```{code-cell} ipython3

# Add an offset (constant column in the design)
X_offset = np.hstack([np.ones_like(neuron_counts)[:, None], X])

# Compute the linear-Gaussian ML estimator
XTX_inv = np.linalg.pinv(X_offset.d[window_size:].T @ X_offset[window_size:])
wsta_offset =  XTX_inv @ (X_offset.d[window_size:].T @ neuron_counts[window_size:]) 

# splitin intercept and coefficients
intercept = wsta_offset[0]
wsta_offset = wsta_offset[1:] # the linear filter part

# Compute prediction with offset
sppred_lgGLM_offset = intercept +  X @ wsta_offset

plt.figure()
markerline,_,_ = plt.stem(neuron_counts.get(*ep_1sec).t, neuron_counts.get(*ep_1sec), linefmt='b-', basefmt='k-', label="spike ct")
plt.setp(markerline, 'markerfacecolor', 'none')
plt.setp(markerline, 'markeredgecolor', 'blue')
plt.plot(sppred_lgGLM.get(*ep_1sec), color='red', linewidth=2, label="lgGLM")
plt.plot(sppred_lgGLM_offset.get(*ep_1sec), color='gold', linewidth=2, label="lgGLM + offset")
plt.title('linear-Gaussian GLM: spike count prediction')
plt.ylabel('spike count'); plt.xlabel('time (s)')
plt.ylim(-1.2, 4)
plt.legend()
plt.show()

# Let's report the relevant training error (squared prediction error on 
# training data) so far just to see how we're doing:
mse1 = np.nanmean((neuron_counts.d - sppred_lgGLM)**2)   # mean squared error, GLM no offset
mse2 = np.nanmean((neuron_counts.d - sppred_lgGLM_offset)**2)  # mean squared error, with offset
rss = np.nanmean((neuron_counts.d - np.mean(neuron_counts))**2)    # squared error of spike train
print('Training perf (R^2): lin-gauss GLM, no offset: {:.2f}'.format(1-mse1/rss))
print('Training perf (R^2): lin-gauss GLM, w/ offset: {:.2f}'.format(1-mse2/rss))
```

## Linear-Gaussian GLM with NeMoS

Fitting a Linear Gaussian GLM with nemos is straightforward, let's see how.

```{code-cell} ipython3

# Define a model object
model = nmo.glm.GLM(observation_model="Gaussian", solver_name="BFGS")
model.fit(X, neuron_counts)
model
```

[//]: # (TODO: drop the solver_name parameter once the Newton PR is done)
As we can see, all we had to set is the observation model to `Gaussian`, default would be `Poisson`. Since the inverse link function is the identity by default, the model we just fit is a linear regression.

Let's plot the predictions and compare the resulting coefficients with the one obtained via the analytical formula.

```{code-cell} ipython3

sppred_lgGLM_nemos = model.predict(X)

plt.figure()
markerline,_,_ = plt.stem(neuron_counts.get(*ep_1sec).t, neuron_counts.get(*ep_1sec), linefmt='b-', basefmt='k-', label="spike ct")
plt.setp(markerline, 'markerfacecolor', 'none')
plt.setp(markerline, 'markeredgecolor', 'blue')
plt.plot(sppred_lgGLM.get(*ep_1sec), color='red', linewidth=2, label="lgGLM")
plt.plot(sppred_lgGLM_offset.get(*ep_1sec), color='gold', linewidth=2, label="lgGLM + offset")
plt.plot(sppred_lgGLM_nemos.get(*ep_1sec), "--k", label="lgGLM nemos")
plt.title('linear-Gaussian GLM: spike count prediction')
plt.ylabel('spike count'); plt.xlabel('time (s)')
plt.ylim(-1.2, 4)
plt.legend()
plt.show()

mse_nemos = np.nanmean((neuron_counts.d - sppred_lgGLM_nemos)**2)
print('Training perf (R^2): lin-gauss GLM, nemos: {:.2f}'.format(1-mse_nemos/rss))
```

## Poisson GLM

Fitting a poisson GLM with exponential non-linearity is actually as easy as a linear regression, let's do it.

```{code-cell} ipython3
# Instantiate a Poisson GLM (or change the observation model, `
# model.observation_model = "Poisson"`)
poisson_model = nmo.glm.GLM(solver_name="BFGS").fit(X, neuron_counts)
poisson_model
```

And computing the predicted rate is more of the same.

```{code-cell} ipython3
rate_pred_pGLM = poisson_model.predict(X)


fig, (ax1,ax2) = plt.subplots(2, figsize=(8, 6))

ax1.plot(ttk, model.coef_/np.linalg.norm(model.coef_), 'o-', label='lin-gauss GLM filt', c='gold')
ax1.plot(ttk, poisson_model.coef_/np.linalg.norm(poisson_model.coef_), 'o-', label='poisson GLM filt', c='red')
ax1.legend(loc = 'upper left')
ax1.set_title('(normalized) linear-Gaussian and Poisson GLM filter estimates')
ax1.set_xlabel('time before spike (s)')
ax1.set_xlim([ttk[0], ttk[-1]])

markerline,stemlines,baseline = plt.stem(neuron_counts.get(*ep_1sec).t, neuron_counts.get(*ep_1sec), linefmt='b-', basefmt='k-', label="spike ct")
plt.setp(markerline, 'markerfacecolor', 'none')
plt.setp(stemlines, color='b', linewidth=.5)
plt.setp(baseline, color='b', linewidth=.5)
ax2.plot(sppred_lgGLM_nemos.get(*ep_1sec), color='gold', linewidth=2, label="lgGLM + offset")
ax2.plot(rate_pred_pGLM.get(*ep_1sec), label="exp-poisson GLM", c='red') 
ax2.set_title('spike count / rate predictions')
ax2.set_ylabel('spike count / bin'); plt.xlabel('time (s)')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()
```



## Non-parametric estimate of the nonlinearity

A way to estimate the non-linearity from the data is computing the filtered stimulus (the rate before the non-linearity is applied), bin it over the range, and compute the mean firing rate per each bin. What I just described is just a tuning curve, and `pynapple` has built-in functions for this.


```{code-cell} ipython3
from copy import deepcopy

# first let's copy our poisson GLM, to keep the original unchanged
cp_poisson_model = deepcopy(poisson_model)

# replace the exp non-linearity with the identity
cp_poisson_model.inverse_link_function = lambda x: x

# this compute identity(X @ model.coef_ + model.intercept_)
raw_filter_output = cp_poisson_model.predict(X)

# compute the tuning curve (the output is an xarray, the)
tc = nap.compute_tuning_curves(units[cell_idx], raw_filter_output.dropna(), bins=25, feature_names=["linpred"])

tc.plot()
plt.show()

```

Let's convert our tuning curve to a function by linear interpolation using `scipy.interp1d`.

```{code-cell} ipython3
from scipy.interpolate import interp1d

fnlin = interp1d(tc.linpred.values, tc.values[0], kind='nearest', bounds_error=False, fill_value='extrapolate')

# Plot exponential and nonparametric nonlinearity estimate
fig, ax = plt.subplots(1, figsize=(10,4)) 
x = np.linspace(tc.linpred.values[0], tc.linpred.values[-1], 100)
ax.plot(x, np.exp(x) * neuron_counts.rate, label='exponential f', c='b')
ax.plot(x, fnlin(x), label='nonparametric f', c='orange')
ax.set_xlabel('filter output')
ax.set_ylabel('rate (sp/s)')
ax.legend(loc='upper left')
ax.set_title('nonlinearity')
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3

# comptue the **total** model log-likelihood 
# default aggregation would be `np.mean`, giving a likelihood per-sample
ll_exp_pglm = poisson_model.score(X, neuron_counts, aggregate_sample_scores=np.sum)

# Now compute the rate under "homogeneous" Poisson model that assumes a
# constant firing rate with the correct mean spike count.
valid_counts = neuron_counts[window_size:].d
ll0 = model.observation_model.log_likelihood(
    valid_counts, 
    np.mean(valid_counts) * np.ones_like(valid_counts),
    aggregate_sample_scores=np.sum
)

LL_expGLM = np.nansum(neuron_counts * np.log(rate_pred_pGLM)) - np.nansum(rate_pred_pGLM)


```