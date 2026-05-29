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

## Building the Design Matrix

Now it's time to create design matrix of our model. What we want as a predictor is the stimulus history over a fixed size window $w$. What we want is to predict is the spike counts at time $t$, $y_t$ from the stimulus at times $s_{t-1},\dots, s_{t-w}$. 

The resulting design matrix will look like this,

$$
X = \begin{bmatrix}
s_{0} & s_1 & \dots & s_{w} \\
s_{1} & s_2 & \dots & s_{w+1} \\
\dots & \dots & \dots & \dots \\
s_{T-w} & s_2 & \dots & s_{T} \\
\end{bmatrix} \label{eq:design-matrix}
$$

Two things to notice: 1) $X$ has $T-w$ rows, where $T$ is `len(stimulus)`. This happens because we need at least $w$ stimulus values to build a row of design matrix. 2) each row is a shifted copy of the row above. 

A convenient way to construct this design matrix is by convolving the stimulus with an identity matrix and reverse the column order. In `NeMoS`, the convolution with the identity can be applied via the `HistoryConv` basis. 


```{code-cell} ipython3
import nemos as nmo

# Match the original notebook
window_size = 25

# define the basis object
bas = nmo.basis.HistoryConv(25, conv_kwargs={"shift": False})

# Convolve with the identity
X = bas.compute_features(stimulus)
# Reverse column order (to match original notebook)
X = X[:,::-1]
```

As you can see:

1. The design matrix is a `pynapple` object, i.e. the information about the time series is preserved, including teh time axis and the time support.
2. The number of samples in `X` matches that of the `stimulus`. This happens because `NeMoS` NaN pads a convolution in mode valid, which outputs $T - w$ samples. This is convenient because `X` and `counts` remains aligned.

Note that reversing or not the column order results in equivalent designs, however changes how the columns are interpreted: if follow $~\eqref{eq:design-matrix}$, then the first column holds the stimulus values $w$ samples in the past, and the last column holds the stimulus at the current sample.

```{code-cell} ipython3

plt.pcolormesh(X[window_size:window_size+50], cmap="Pastel1")
plt.show()
```

## Compute and visualize the spike-triggered average (STA)

When the stimulus is Gaussian white noise, the STA provides an unbiased estimator for the filter in a GLM / LNP model (as long as the nonlinearity results in an STA whose expectation is not zero; feel free to ignore this parenthetical remark if you're not interested in technical details. It just means that if the nonlinearity is symmetric, eg. x^2, then this condition won't hold, and the STA won't be useful).

In many cases it's useful to visualize the STA (even if your stimuli are not white noise), just because if we don't see any kind of structure then this may indicate that we have a problem (e.g., a mismatch between the design matrix and binned spike counts.


```{code-cell} ipython3
neuron_count = counts[:, cell_idx]

# skip nans
sta = (X.d[window_size:].T @ neuron_count[window_size:]) / neuron_count.sum()

ttk = np.arange(-window_size+1,1) / neuron_count.rate  # time bins for STA (in seconds)
plt.clf()
plt.figure(figsize=[12,8])
plt.plot(ttk,ttk*0, 'k--')
plt.plot(ttk, sta, 'bo-')
plt.title('STA')
plt.xlabel('time before spike (s)')
plt.xlim([ttk[0],ttk[-1]])
plt.show()
```