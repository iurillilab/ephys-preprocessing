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
import ocplot as ocp

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
def get_trace_snippets(rec, sample_every_n_channels=1, snippet_length_ms=3000, 
                       n_snippets=5, offset_subtract=False, verbose=True):
    """Get random snippets of traces from recording, sampled across channels."""
    channels_to_sample = rec.get_channel_ids()[::sample_every_n_channels]
    
    snippet_length_samples = int(snippet_length_ms * 0.001 * rec.get_sampling_frequency())
    n_samples = rec.get_total_samples()

    trace_snippets = []
    onsets = sorted(np.random.randint(0, n_samples - snippet_length_samples, n_snippets))
    pbar = tqdm(onsets, desc="Getting trace snippets: ") if verbose else onsets
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

def compute_power_spectrum(traces, fs, freq_range=(10, 10000)):
    """Compute power spectrum using Welch's method and average across channels"""
    freqs = np.arange(freq_range[0], freq_range[1], 1)
    nperseg = fs
    noverlap = fs // 2
    
    f, pxx = signal.welch(traces, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum', axis=0)

    # find closest frequency indices
    freq_mask = (f >= freqs[0]) & (f <= freqs[-1])
    f = f[freq_mask]
    pxx = pxx[freq_mask]
    
    return f, pxx

def compute_voltage_histograms(traces, hist_bins, verbose=True):
    """Compute voltage histograms for each channel in traces."""
    hist_counts = []
    pbar = tqdm(range(traces.shape[1]), desc="Computing histograms: ") if verbose else range(traces.shape[1])
    for channel_idx in pbar:
        hist_counts.append(np.histogram(traces[:, channel_idx], bins=hist_bins, density=True)[0])
    return np.array(hist_counts).T


def plot_noise_analysis(traces, hist_counts, f, pxx, snippet_length_ms, fs, vlim=10, cbar_fract=0.33, ratio=0.5):
    """Create a figure showing noise analysis results with traces, histograms and power spectra."""
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                         hspace=0.5, wspace=0.4)

    # Top panel - traces
    ax_traces = fig.add_subplot(gs[0, :])
    im_traces = ax_traces.imshow(traces.T[:,:int(2*snippet_length_ms*fs*0.001)], 
                                aspect="auto",
                                extent=(0, 2*snippet_length_ms*0.001, 0, traces.shape[1]),
                                origin="lower", vmin=-vlim, vmax=vlim, cmap="gray")
    # mark snippet borders with a line:
    for n in range(2):
        ax_traces.axvline(n * snippet_length_ms * 0.001, c="r", lw=0.5)
    ax_traces.set_xlabel("Time (s)")
    ax_traces.set_ylabel("Channel")

    cbar_height = cbar_fract
    cax_traces = ax_traces.inset_axes([1.02, 0.5-cbar_height/2, 0.02, cbar_height])
    plt.colorbar(im_traces, cax=cax_traces, orientation='vertical', label="Voltage (uV)", ticks=[-vlim, 0, vlim])

    ####
    #  Bottom left - histogram
    gs_hist = gs[1, 0].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.1)
    ax_hist_avg = fig.add_subplot(gs_hist[0])
    hist_bins = np.arange(-50, 51, 5)  # Moved from global scope
    ax_hist_avg.plot(hist_bins[:-1], hist_counts.mean(axis=1), 'k-')
    ax_hist_avg.set_ylabel("Mean density")
    # plot lines for 1sigma:
    sigma = np.std(traces.flatten())
    ax_hist_avg.axvline(2*sigma, c='darkblue', ls='--', lw=0.5, label='±2σ')
    ax_hist_avg.axvline(-2*sigma, c='darkblue', ls='--', lw=0.5)
    ax_hist_avg.legend(frameon=False)
    
    ocp.despine(ax_hist_avg)

    ax_hist = fig.add_subplot(gs_hist[1], sharex=ax_hist_avg)
    im_hist = ax_hist.imshow(hist_counts.T, aspect="auto", origin="lower",
                            extent=(hist_bins[0], hist_bins[-1], 0, hist_counts.shape[1]))
    ax_hist.set_xlabel("Voltage (uV)")
    ax_hist.set_ylabel("Channel")
    ax_hist.set_xticks(np.arange(-48, 51, 12))

    cax_hist = ax_hist.inset_axes([1.02, 0.5-cbar_height/2, 0.02, cbar_height])
    plt.colorbar(im_hist, cax=cax_hist, orientation='vertical', label="Density", ticks=[])

    ####
    #  Bottom right - power spectrum
    gs_power = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.1)
    ax_power_avg = fig.add_subplot(gs_power[0])
    ax_power_avg.loglog(f, pxx.mean(axis=1), 'k-', zorder=100)
    ax_power_avg.set_ylabel("Mean power")

    for ref_freq in [50, 150]:
        ax_power_avg.axvline(ref_freq, c="k", ls="--", lw=0.5)
        ax_power_avg.text(ref_freq+5, ax_power_avg.get_ylim()[1]*0.7, f"{ref_freq} Hz", c="k", fontsize=6, zorder=-100)
    ax_power_avg.axvspan(300, 1000, color=".9", zorder=-1000)
    ocp.despine(ax_power_avg)

    ax_power = fig.add_subplot(gs_power[1], sharex=ax_power_avg)
    im_power = ax_power.imshow(np.log10(pxx).T, aspect="auto", origin="lower",
                              extent=(f[0], f[-1], 0, pxx.shape[1]))
    ax_power.set(xlabel="Frequency (Hz)")
    cax_power = ax_power.inset_axes([1.02, 0.5-cbar_height/2, 0.02, cbar_height])
    plt.colorbar(im_power, cax=cax_power, orientation='vertical', label="Log power", ticks=[])

    return fig

# %%
n_snippets = 5
snippet_length_ms = 3000
sample_every_n_channels = 1
hist_bins = np.arange(-50, 51, 5)
fs = rec.get_sampling_frequency()

rec = rec2
traces = get_trace_snippets(rec, sample_every_n_channels=sample_every_n_channels, 
                           snippet_length_ms=snippet_length_ms, n_snippets=n_snippets)

hist_counts = compute_voltage_histograms(traces, hist_bins)
f, pxx = compute_power_spectrum(traces, rec, fs)
fig = plot_noise_analysis(traces, hist_counts, f, pxx, snippet_length_ms, fs,)

# %%

plt.show()
# %%

# %%
# Set logarithmic scale
ax.set_xscale('log')
# Add colorbar and labels
cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label="Log power")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Channel")

plt.show()
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
