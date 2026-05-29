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

This tutorial is an adaptation of [JW Pillow](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/tutorial1_PoissonGLM.ipynb)'s matherial presented at a short course on 

This is a tutorial illustrating the fitting of a linear-Gaussian GLM (also known as linear least-squares regression model) and a Poisson GLM (aka  "linear-nonlinear-Poisson" model) to retinal ganglion cell (RGC) spike trains stimulated with binary temporal white noise. 

(Data from [Uzzell & Chichilnisky, 2004](http://jn.physiology.org/content/92/2/780.long); see [`README.txt`](https://github.com/pillowlab/GLMspiketraintutorial_python/blob/main/data_RGCs/README.txt) for details).

The dataset is provided for tutorial purposes only, and should not be distributed or used for publication without express permission from EJ Chichilnisky (ej@stanford.edu).

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


### Spike Times as TsGroup

Let's start by loading the spike times. This will be loaded as a list of 4 arrays, each containing the spike trains of a recorded unit.

```{code-cell} ipython3
from scipy.io import loadmat

# Load the array of spike times (one array per unit, 4 units total)
spike_times = loadmat(data_paths["SpTimes.mat"], simplify_cells=True)["SpTimes"]

print("Number units: ", len(spike_times))

# Print the spike times of the first unit.
print("Spike times of unit 0:")
print(spike_times[0])
```

A list of spike times can be loaded directly into a `pynapple` [TsGroup](https://pynapple.org/generated/pynapple.TsGroup.html), a dictionary object conaining multiple timestamps arrays, and associated metadata if present.

```{code-cell} ipython3
import pynapple as nap

# Load spike times into pynapple
units = nap.TsGroup({i: nap.Ts(val) for i, val in enumerate(spike_times)})
units
```

The [TsGroup](https://pynapple.org/generated/pynapple.TsGroup.html) is represented as a table, the first column is the unit index. If not provided 0, ..., num units - 1 will be used. The second column is the mean firing rate of the unit in Hz, computed directly from the spike trains.  When present, additional columns include the associated metadata.


### Stimulus as a Tsd

For this dataset, the stimulus is a full-field binary white noise. Stimulus presentaiton time and stimulus values are stored in separate 1-D arrays.

```{code-cell} ipython3
# Load stimulus times and values
stim_times = loadmat(data_paths["stimtimes.mat"], simplify_cells=True)["stimtimes"]
stim = loadmat(data_paths["Stim.mat"], simplify_cells=True)["Stim"]

print(f"\ntimes: {stim_times[:5]}\n\nvalues: {stim[:5]}")
```

`pynapple` stores 1-D time series with data as `Tsd` objects. These objects have a time attribute `t` containing the time stamps, and a data attribute `d` containing the data.

```{code-cell} ipython3
stimulus = nap.Tsd(stim_times, stim)

print(f"\ntimes: {stimulus.t[:5]}\n\nvalues: {stimulus.d[:5]}")
```

Finally, let's plot the 
## Pre-processing

After loading the two time series in `pynapple` we would like to pre-process the activity by binning the spikes
Additionally, and very conveniently, every time series object in `pynapple` stores a `time_support` attribute, which of [`IntervalSet`](https://pynapple.org/generated/pynapple.IntervalSet.html) set type. IntervalSets  specifies starts and ends of (irregular) continuous recoding epochs.

The time support can be provided at object initialization. If not provided, the support is inferred from data, as a single epoch from the minimum to the maximum of the provided time stamps.

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

```{code-cell} ipython3

```
