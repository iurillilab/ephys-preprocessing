# %%
# %matplotlib widget
import spikeinterface.extractors as se
from spikeinterface.core import get_noise_levels
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nwb_conv.oephys import OEPhysDataFolder
from scipy import signal

# %%
# Path to a valid OpenEphys recording:
# main_dir = Path("/Users/vigji/Desktop/noise_tests")
main_dir = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/setup-calibrations/ephys-rigs-tests")
assert main_dir.exists(), "main_dir does not exist, check path"
date = "20241115"


def get_recording(setup, probe_combination, probe_name):
    setup = main_dir / setup / date
    probe_dir = setup / f"{probe_combination}_test_noise"
    # print(probe_dir, probe_dir.exists())
    recording_path = next(probe_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"))
    rec_folder = OEPhysDataFolder(recording_path)
    # print(rec_folder.reference_stream_name)
    probe_stream_name = [stream for stream in rec_folder.stream_names if probe_name in stream][0]
    print(probe_stream_name)
    recording = se.read_openephys(recording_path, stream_name=probe_stream_name)
    return recording

def get_recording_noise_levels(setup, probe_combination, probe_name, method="std"):
    recording = get_recording(setup, probe_combination, probe_name)
    return get_noise_levels(recording, method=method)


# %%
setup1, key1 = "mpm-rig", "NPX1"
setup2, key2 = "ephysroom-rig", "NPX1"
rec1 = get_recording(setup1, key1, "ProbeA")
rec2 = get_recording(setup2, key2, "ProbeA")
# noise_levs1 = get_recording_noise_levels(setup1, key1, "ProbeB")
# noise_levs2 = get_recording_noise_levels(setup2, key2, "ProbeA")

# %%
rec = rec2
def get_trace_snippets(rec, sample_every_n_channels=1, snippet_length_ms=3000, n_snippets=5, offset_subtract=False):
    """Get random snippets of traces from recording, sampled across channels."""
    channels_to_sample = rec.get_channel_ids()[::sample_every_n_channels]
    
    snippet_length_samples = int(snippet_length_ms * 0.001 * rec.get_sampling_frequency())
    n_samples = rec.get_total_samples()

    trace_snippets = []
    onsets = sorted(np.random.randint(0, n_samples - snippet_length_samples, n_snippets))
    pbar = tqdm(onsets, desc="Getting trace snippets: ")
    for starting_sample in pbar:
        trace_snippet = np.array(
            rec.get_traces(
                start_frame=starting_sample,
                end_frame=starting_sample + snippet_length_samples,
                channel_ids=channels_to_sample,
                return_scaled=True,
            )
        )
        if offset_subtract:
            trace_snippet = trace_snippet - np.median(trace_snippet, axis=0)
        trace_snippets.append(trace_snippet)

    return np.concatenate(trace_snippets, axis=0)

def compute_power_spectrum(traces, rec, freq_range=(10, 10000)):
    """Compute power spectrum using Welch's method and average across channels"""
    freqs = np.arange(freq_range[0], freq_range[1], 1)
    fs = rec.get_sampling_frequency()
    nperseg = fs
    noverlap = fs // 2
    
    f, pxx = signal.welch(traces, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum', axis=0)

    # find closest frequency indices
    freq_mask = (f >= freqs[0]) & (f <= freqs[-1])
    f = f[freq_mask]
    pxx = pxx[freq_mask]

    # average across channels
    mean_pxx = np.mean(pxx, axis=1)
    
    return f, mean_pxx

def compute_voltage_histograms(traces, hist_bins):
    """Compute voltage histograms for each channel in traces."""
    hist_counts = []
    for channel_idx in tqdm(range(traces.shape[1]), desc="Computing histograms: "):
        hist_counts.append(np.histogram(traces[:, channel_idx], bins=hist_bins, density=True)[0])
    return np.array(hist_counts).T

n_snippets = 5
snippet_length_ms = 3000
sample_every_n_channels = 1
hist_bins = np.arange(-50, 51, 5)
traces = get_trace_snippets(rec, sample_every_n_channels=sample_every_n_channels, 
                           snippet_length_ms=snippet_length_ms, n_snippets=n_snippets)

hist_counts = compute_voltage_histograms(traces, hist_bins)
f, mean_pxx = compute_power_spectrum(traces, rec)
# %%
traces.shape
# %%
vlim = 10
cbar_fract = 0.33
ratio = .3
fig, axs = plt.subplots(1,2, figsize=(10, 3), sharey=True, 
                        gridspec_kw={"top": 0.8, "width_ratios": [1, ratio]})

ax = axs[0]
im = ax.imshow(traces.T[:,:int(2*snippet_length_ms*rec.get_sampling_frequency()*0.001)], aspect="auto",
           extent=(0, 2*snippet_length_ms*0.001, 0, traces.shape[1]),
           origin="lower", vmin=-vlim, vmax=vlim, cmap="gray")
# mark snippet borders with a line:
for n in range(2):
    ax.axvline(n * snippet_length_ms * 0.001, c="r", lw=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Channel")
cax = ax.inset_axes([1-cbar_fract*ratio, 1.15, cbar_fract*ratio, 0.05])
plt.colorbar(im, cax=cax, orientation='horizontal', label="Voltage (uV)", ticks=[-vlim, 0, vlim])
cax.xaxis.set_ticks_position('top')

ax = axs[1]
im = ax.imshow(hist_counts.T, aspect="auto", origin="lower", 
          extent=(hist_bins[0], hist_bins[-1], 0, hist_counts.shape[1]))
ax.set_xlabel("Voltage (uV)")
ax.set_xticks(np.arange(-50, 51, 25))

# Add colorbar
cax = ax.inset_axes([1-cbar_fract, 1.15, cbar_fract, 0.05])
plt.colorbar(im, cax=cax, orientation='horizontal', label="Density", ticks=[])

# cax.xaxis.set_label_position('top')

# %%
# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax.loglog(f, mean_pxx, zorder=100)
for ref_freq in [50, 150]:
    plt.axvline(ref_freq, c="k", ls="--", lw=0.5)
    ax.text(ref_freq+5, ax.get_ylim()[1]*0.7, f"{ref_freq} Hz", c="k", fontsize=6, zorder=-100)
ax.axvspan(300, 1000, color=".9", zorder=-1000)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
#plt.grid(True)

# %%

conditions = [
    (
        ("ephysroom-rig", "NPX1-NPX2"),
        ("ephysroom-rig", "NPX1-NPX2_light")
    ),
    (
        ("ephysroom-rig", "NPX1-NPX2"), 
        ("ephysroom-rig", "NPX1-NPX2_light_cover")
    )
]

for (setup1, key1), (setup2, key2) in conditions:
    noise_levs1 = get_recording_noise_levels(setup1, key1, "ProbeA")
    noise_levs2 = get_recording_noise_levels(setup2, key2, "ProbeA")

    f, ax = plt.subplots(figsize=(3, 3), gridspec_kw={"top": 0.9, "bottom"=0.2,
                                                      "left": 0.2, "right": 0.2})
    ax.scatter(noise_levs1, noise_levs2, s=10, alpha=0.5, lw=0)
    xlims = (0, max(np.max(noise_levs1), np.max(noise_levs2))*1.02)
    ax.plot(*[[1, xlims[1]*0.9]]*2, "k--", lw=1)
    ax.set_xlim(xlims)
    ax.set_ylim(xlims)
    ax.set_xlabel(f"{setup1} {key1}")
    ax.set_ylabel(f"{setup2} {key2}")
    ax.set_yticks(ax.get_xticks())
    plt.tight_layout()
# %%
