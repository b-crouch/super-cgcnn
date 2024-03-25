from mp_api.client import MPRester
import pandas as pd
import numpy as np
import pymatgen
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
import emmet
from emmet.core.summary import HasProps
from emmet.core.mpid import MPID
import scipy.interpolate as inter
from itertools import cycle 
import pickle

def get_bands(bs, plot=False):
    """
    Extracts band data – k-point distances, energies, spin states – from pymatgen BandStructure object
    Args:
        bs (BandStructureSymmLine): band structure object
        plot (bool): if True, outputs axis tick data needed for plotting band structure
    Returns:
        all_distances (np.array): k-point distances of points after B-spline interpolation
        all_energies (np.array): band energies of points after B-spline interpolation (scaled such that E_f = 0)
        all_spins (np.array): spin state of each band
    """
    bs_helper = BSPlotter(bs)
    data = bs_helper.bs_plot_data(zero_to_efermi=True) # All bands are expressed relative to E_f = 0
    distances = data["distances"] # Distance scale for MP band structure k-points
    all_energies, all_distances, all_spins = None, None, None
    for spin in bs.bands:
        energies = data["energy"][str(spin)]
        interp_distances, interp_energies = bs_helper._interpolate_bands(distances, energies)
        if all_energies is None:
            all_distances, all_energies = np.hstack(interp_distances), np.hstack(interp_energies)
            all_spins = int(spin)*np.ones(len(all_energies))
        else:
            all_energies = np.vstack([all_energies, np.hstack(interp_energies)])
            all_spins = np.hstack([all_spins, int(spin)*np.ones(len(all_energies))])
    if plot:
        return all_distances, all_energies, all_spins, (data["ticks"]["distance"], data["ticks"]["label"])
    else:
        return all_distances, all_energies, all_spins

def plot_bands(distances, energies, spins, tick_data, annotate=None, ylim=(-1.5, 1.5), title=None, img_dest=None):
    """
    Plots band structure with bands of interest highlighted
    Args:
        distances (np.array): k-point distances of points on each band
        energies (np.array): energies of points on each band
        spins (np.array): spin state of each band
        tick_data (tuple of lists): tick labels for horizontal and vertical axes
        annotate (list of tuples): list of (idx, label) pairs for each band to highlight/label in plot
        ylim (tuple of floats): lower and upper bound of vertical axis to plot
        title (string): title of plot
        img_dest (string): filepath for saving plot
    """
    if annotate:
        indices = [band[0] for band in annotate]
        labels = [band[1] for band in annotate]
        annot_colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
        for i in range(2):
            next(annot_colors) #I like starting with red for visualization
    fig = plt.figure()
    ax = plt.subplot()
    for i, band in enumerate(energies):
        if not annotate or i not in indices:
            ax.plot(distances, band, c="tab:blue", linestyle="-" if spins[i]==1 else "--")
        elif annotate and i in indices:
            ax.plot(distances, band, label=labels[indices.index(i)], c=next(annot_colors), linestyle="-" if spins[i]==1 else "--")
    ax.set_ylim(*ylim)
    ax.set_xlim(distances[0], distances[-1])
    ax.set_xticks(tick_data[0])
    ax.set_xticklabels(tick_data[1])
    ax.set_title(f"{title if title else ''}")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    if annotate:
        plt.legend(loc="lower right")
    if img_dest:
        fig.savefig(img_dest)
    plt.close()

def near_fermi(energies, fermi_window):
    """
    For array of band energies, returns indices of bands that pass within +-`fermi_window` of the Fermi level
    """
    return np.where(np.any((energies >= -fermi_window) & (energies <= fermi_window), axis=1))

def intersects_fermi(energies):
    """
    For array of band energies, returns indices of bands that intersect the Fermi level
    """
    return np.any(energies > 0, axis=1)&np.any(energies < 0, axis=1)

def compute_flatness(energies):
    """
    Compute flatness score of bands stored in `energies`, defined as the standard deviation of residuals from the mean band energy
    """
    return np.std(energies, axis=1)

def compute_bandwidth(energies):
    """
    Compute bandwidths of bands stored in `energies`, defined as the maximum energy of band minus minimum energy of band
    """
    return np.max(energies, axis=1) - np.min(energies, axis=1)

def score_variation(scores):
    """
    Compute standard deviation of flatness metrics (flatness or bandwidth)
    """
    return np.std(scores)

def score_range(scores):
    """
    Compute range of flatness metrics (flatness or bandwidth)
    """
    return np.max(scores) - np.min(scores)

def is_flat_steep_system(energies, fermi_window, metric, threshold_percent):
    """
    Detects the presence of a flat-steep system, defined by Deng et al. as a band structure where 1) a flat band lies within a narrow energy window near E_f
    and 2) a dispersive band intersects the Fermi level
    Args:
        energies (np.array): energies of points on each band
        fermi_window (float): deviation from E_f considered "near Fermi"
        metric (string): flatness metric to use when screening for flat band (flatness or bandwidth)
        threshold_percent (float): band is considered "flat" if its flatness is <=threshold_percent * mean flatness of system
    Returns:
        True if such a flat-steep system exists, otherwise False
    """
    assert metric == "flatness" or metric == "bandwidth", "Allowable scoring metrics are 'flatness' or 'bandwidth'"
    if metric == "flatness":
        all_scores = compute_flatness(energies)
        near_fermi_idx = near_fermi(energies, fermi_window)
        bands_near_fermi = energies[near_fermi_idx]
        window_scores = compute_flatness(bands_near_fermi)
    elif metric == "bandwidth":
        all_scores = compute_bandwidth(energies)
        near_fermi_idx = near_fermi(energies, fermi_window)
        bands_near_fermi = energies[near_fermi_idx]
        window_scores = compute_bandwidth(bands_near_fermi)
    flat_threshold = threshold_percent*np.mean(all_scores)
    has_flat_band = np.any(window_scores <= flat_threshold)
    has_dispersive_band = np.any(intersects_fermi(energies[all_scores > flat_threshold])) # Flat band can't count as the dispersive band!
    return has_flat_band and has_dispersive_band

def query_band_structure(mpid, api_key):
    """
    Queries BandStructureSymmLine object from Materials Project
    Args:
        mpid (string): Materials Project ID of desired structure
        api_key (string): Materials Project API key
    Returns:
        bs (BandStructureSymmLine): structure object of queried band structure
        dct (dict): dictionary of band structure data for saving to local file
    """
    with MPRester(api_key) as mpr:
        bs = mpr.get_bandstructure_by_material_id(mpid)  

    # Adapted from pymatgen repo
    dct = {
        "@module": type(bs).__module__,
        "@class": type(bs).__name__,
        "lattice_rec": bs.lattice_rec.as_dict(),
        "efermi": bs.efermi,
        "kpoints": [],
    }
    for k in bs.kpoints:
        dct["kpoints"].append(k.as_dict()["fcoords"])

    dct["bands"] = {str(int(spin)): bs.bands[spin].tolist() for spin in bs.bands}
    dct["is_metal"] = bs.is_metal()
    vbm = bs.get_vbm()
    dct["vbm"] = {
        "energy": vbm["energy"],
        "kpoint_index": vbm["kpoint_index"],
        "band_index": {str(int(spin)): vbm["band_index"][spin] for spin in vbm["band_index"]},
        "projections": {str(spin): v.tolist() for spin, v in vbm["projections"].items()},
    }
    cbm = bs.get_cbm()
    dct["cbm"] = {
        "energy": cbm["energy"],
        "kpoint_index": cbm["kpoint_index"],
        "band_index": {str(int(spin)): cbm["band_index"][spin] for spin in cbm["band_index"]},
        "projections": {str(spin): v.tolist() for spin, v in cbm["projections"].items()},
    }
    dct["band_gap"] = bs.get_band_gap()
    dct["labels_dict"] = {}
    dct["is_spin_polarized"] = bs.is_spin_polarized

    for c, label in bs.labels_dict.items():
        mongo_key = c if not c.startswith("$") else f" {c}"
        dct["labels_dict"][mongo_key] = label.as_dict()["fcoords"]
    dct["projections"] = {}
    if len(bs.projections) != 0:
        dct["structure"] = bs.structure.as_dict()
        dct["projections"] = {str(int(spin)): np.array(v).tolist() for spin, v in bs.projections.items()}
    return bs, dct

def characterize_bands(mpid, api_key, fermi_windows, plot_window, img_dest=None, save_bs=False):
    """
    Computes flatness metrics and generates band structure plot for structure of interest
    Two definitions of "flatness" are considered. Absolute flatness directly computes the flatness score and bandwidth of each structure without additional normalization.
    Relative flatness scales all flatness scores and bandwidths against the mean value of that metric for the band structure.
    For each combination of absolute/relative flatness/bandwidth, compute the mean band value, flattest band value, standard deviation of values, range of values
    Args:
        mpid (string): Materials Project ID of desired structure
        api_key (string): Materials Project API key
        fermi_windows (list): list of energy values relative to E_f to treat as "near Fermi" (flatness analysis will be rerun for each window size)
        plot_window (float): indiciates which energy range in `fermi_window` to treat as "near Fermi" when plotting band structure
        img_dest (string): filepath for saving band structure plot
        save_bs (bool): if True, saves BandStructureSymmLine object queried from Materials Project
    Returns:
        data (np.array): flatness metrics 
    """
    if save_bs:
        bs, band_dict = query_band_structure(mpid, api_key)
    with MPRester(api_key) as mpr:
        # If full BandStructure object not needed for saving, MPR helper runs the query more quickly 
        bs = mpr.get_bandstructure_by_material_id(mpid) 
        band_dict = {}
    assert plot_window in fermi_windows, "`plot_window` must be one of the `fermi_windows` being analyzed!"
    plot_idx = fermi_windows.index(plot_window)
    k_point_distances, band_energies, spins, ticks = get_bands(bs, plot=True)
    data = np.zeros(18*len(fermi_windows))
    for i, fermi_window in enumerate(fermi_windows):
        fermi_window_idx = near_fermi(band_energies, fermi_window)
        fermi_window_energies, fermi_window_spins = band_energies[fermi_window_idx], spins[fermi_window_idx]
        if len(fermi_window_energies) == 0:
            # No bands lie in Fermi window; fill flatness metrics with inf
            data[18*i:18*(i+1)] = [np.inf]*18
            continue
        # Compute flatness scores
        near_fermi_flatnesses = compute_flatness(fermi_window_energies)
        is_flat_steep_flatness = is_flat_steep_system(band_energies, fermi_window, "flatness", 0.2)
        # Apply the absolute metric for flatness scores
        mean_flatness, sd_flatness, range_flatness, min_flatness = np.mean(near_fermi_flatnesses), score_variation(near_fermi_flatnesses), score_range(near_fermi_flatnesses), np.min(near_fermi_flatnesses)
        # Apply the relative metric for flatness scores
        scaled_flatness = near_fermi_flatnesses/np.mean(compute_flatness(band_energies))
        mean_relative_flatness, sd_relative_flatness, range_relative_flatness, min_relative_flatness = np.mean(scaled_flatness), score_variation(scaled_flatness), score_range(scaled_flatness), np.min(scaled_flatness)
       
        # Compute bandwidth scores
        near_fermi_bandwidths = compute_bandwidth(fermi_window_energies)
        is_flat_steep_bandwidth = is_flat_steep_system(band_energies, fermi_window, "bandwidth", 0.2)
        mean_bandwidth, sd_bandwidth, range_bandwidth, min_bandwidth = np.mean(near_fermi_bandwidths), np.std(near_fermi_bandwidths), score_range(near_fermi_bandwidths), np.min(near_fermi_bandwidths)
        scaled_bandwidth = near_fermi_bandwidths/np.mean(compute_bandwidth(band_energies))
        mean_relative_bandwidth, sd_relative_bandwidth, range_relative_bandwidth, min_relative_bandwidth = np.mean(scaled_bandwidth), score_variation(scaled_bandwidth), score_range(scaled_bandwidth), np.min(scaled_bandwidth)

        data[18*i:18*(i+1)] = [min_flatness, min_relative_flatness, mean_flatness, mean_relative_flatness, sd_flatness, sd_relative_flatness, range_flatness, range_relative_flatness, is_flat_steep_flatness, min_bandwidth, \
                               min_relative_bandwidth, mean_bandwidth, mean_relative_bandwidth, sd_bandwidth, sd_relative_bandwidth, range_bandwidth, range_relative_bandwidth, is_flat_steep_bandwidth]
        if img_dest and i == plot_idx:
            flattest_idx = np.argmin(near_fermi_flatnesses)
            if not os.path.exists(img_dest):
                os.makedirs(img_dest)
            plot_bands(k_point_distances, fermi_window_energies, fermi_window_spins, ticks, [(flattest_idx, f"Min flatness: {np.round(min_flatness, 5)}")], title=f"{mpid}", img_dest=f"{img_dest}/{mpid}_{plot_window}_eV_from_fermi.png")
    return data, band_dict

def load_band_structure(filepath):
    """
    Load band structure object from local file
    """
    with open(filepath, "rb") as f:
        loaded_bs = pickle.load(f)
    return BandStructureSymmLine.from_dict(loaded_bs)