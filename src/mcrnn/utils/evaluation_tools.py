"""
This script contains python functions that are useful for model evaluation, such as data writing and plotting.
"""

import os
import subprocess
from typing import List, Dict
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig

def compute_metrics(cfg: DictConfig, ds: xr.Dataset) -> None:
    """
    Compute RMSE and Frobenius Norm (accumulated error) and print them to console.

    :param cfg: The configuration of the model
    :param ds: The dataset containing the model outputs (predictions) and targets
    """

    print("Model name:", cfg.model.name)

    T = ds.sizes["time"]  # Number of time steps
    tf = cfg.testing.teacher_forcing_steps
    diff = ds.outputs-ds.targets

    # Root mean squared error overall, in teacher forcing, and in closed loop
    rmse = str(np.round(np.sqrt(((diff)**2).mean()).values, 4)).ljust(6)
    rmse_tf = str(np.round(np.sqrt(((diff).sel(time=slice(0, tf))**2).mean()).values, 4)).ljust(6)
    rmse_cl = str(np.round(np.sqrt(((diff).sel(time=slice(tf, T))**2).mean()).values, 4)).ljust(6)
    print("RMSE:", rmse, "\tRMSE TF:", rmse_tf, "\tRMSE CL:", rmse_cl)

def generate_mp4(
    cfg: DictConfig,
    ds: xr.Dataset,
    file_path: str
):
    """
    Generates mp4 video visualizing model output, target, and the difference between those.

    :param cfg: The hydra configuration of the model
    :param ds: An xarray dataset containing model inputs, outputs, and targets
    :param file_path: The path to write the video to
    """

    if cfg.verbose: print("Generating frames and a video of the model predictions...")

    frame_path = os.path.join(os.path.dirname(file_path), "frames")
    os.makedirs(frame_path, exist_ok=True)
    inputs, outputs, targets = ds.inputs.values, ds.outputs.values, ds.targets.values

    # Visualize results
    diff_ot = outputs - targets
    diff_io = inputs - outputs
    diffmax_ot = max(abs(np.min(diff_ot[0, :, 0])),
                     abs(np.max(diff_ot[0, :, 0])))
    diffmax_io = max(abs(np.min(diff_io[0, :, 0])),
                     abs(np.max(diff_io[0, :, 0])))
    vmin, vmax = np.min(targets[0, :, 0]), np.max(targets[0, :, 0])
    for t in range(outputs.shape[1]):
        if cfg.verbose:
            mode = "teacher forcing" if t < cfg.testing.teacher_forcing_steps else "closed loop"
            #print(f"{t}/{cfg.testing.sequence_length}", mode)
        fig, ax = plt.subplots(2, 3, figsize=(13, 8))
        
        ax[0][0].imshow(outputs[0, t, 0], origin="lower", vmin=vmin, vmax=vmax)
        ax[0][0].set_title(r"Prediction ($\hat{y}_t$)")

        im1 = ax[0][1].imshow(inputs[0, t, 0], origin="lower", vmin=vmin, vmax=vmax)
        ax[0][1].set_title(r"Input ($x_t$)")
        divider1 = make_axes_locatable(ax[0][1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1, orientation='vertical')

        im2 = ax[0][2].imshow(diff_io[0, t, 0], origin="lower", vmin=-diffmax_io, vmax=diffmax_io, cmap="bwr")
        ax[0][2].set_title(r"Difference to input ($x_t-\hat{y}_t$)")
        divider2 = make_axes_locatable(ax[0][2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2, orientation='vertical')

        im3 = ax[1][1].imshow(targets[0, t, 0], origin="lower", vmin=vmin, vmax=vmax)
        ax[1][1].set_title(r"Ground truth target ($y_t$)")
        divider3 = make_axes_locatable(ax[1][1])
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax3, orientation='vertical')
        #cax1.yaxis.set_ticks_position('left')

        im4 = ax[1][2].imshow(diff_ot[0, t, 0], origin="lower", vmin=-diffmax_ot, vmax=diffmax_ot, cmap="bwr")
        ax[1][2].set_title(r"Difference to target($\hat{y}_t-y_t$)")
        divider4 = make_axes_locatable(ax[1][2])
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax4, orientation='vertical')

        fig.suptitle(f"Time step = {t+1}/{outputs.shape[1]} ({mode})")
        fig.tight_layout()
        fig.savefig(os.path.join(frame_path, f"state_{str(t).zfill(4)}.png"))
        plt.close()

    # Generate a video from the just generated frames with ffmpeg
    subprocess.run(["ffmpeg",
                    "-f", "image2",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-r", "15",
                    "-pattern_type", "glob",
                    "-i", f"{os.path.join(frame_path, '*.png')}",
                    "-vcodec", "libx264",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    file_path])

def plot_rmse_over_time(
    cfg: DictConfig,
    performance_dict: Dict,
    model_names: List = None,
):
    """
    Plot the root mean squared error of all models (averaged over samples, dimensions, height, width) over time.
    """
    #fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    colors = [
        "dodgerblue",
        "darkmagenta",
        "darkorange",
        "firebrick",
        "darkgreen",
        "gold",
        "yellowgreen",
        "slateblue"
    ]

    def plot_data(rmses: np.array, plot_idx: int, model_name: str):
        rmses = np.array(rmses)
        mean, std = np.mean(rmses, axis=0), np.std(rmses, axis=0)
        model_name = model_names[plot_idx] if model_names else model_name
        ax.plot(range(1, len(mean)+1), mean, label=model_name, color=colors[plot_idx])
        ax.fill_between(range(1, len(mean)+1), mean+std, mean-std, color=colors[plot_idx], alpha=0.4)
        # Print statistics in tex-compatible format to console
        f = 1e-2 if "Geopotential" in cfg.data._target_ else 1e+2
        tf_steps = cfg.testing.teacher_forcing_steps
        mean_tf, std_tf = str(np.round(f*np.mean(mean[:tf_steps]), 2)), str(np.round(f*np.mean(std[:tf_steps]), 4))
        mean_cl, std_cl = str(np.round(f*np.mean(mean[tf_steps:]), 2)), str(np.round(f*np.mean(std[tf_steps:]), 4))
        print(f"{model_name} & 5s & ${mean_tf}\pm{std_tf}$ & ${mean_cl}\pm{std_cl}$\\\\")
        plot_idx += 1

    performance_dict["xxx"] = None  # Dummy entry helper for statistics plotting
    rmse_max = -np.infty
    rmses = []  # List to store rmses of same models trained with different seeds
    plot_idx = 0
    dict_key_list = list(performance_dict.keys())
    for m_idx, model_name in enumerate(dict_key_list[:-1]):
        ds = performance_dict[model_name]
        next_model_name = dict_key_list[m_idx+1]
        rmse = np.sqrt(((ds.outputs-ds.targets)**2).mean(dim=["sample", "dim_out", "height", "width"]))
        rmses.append(rmse)
        rmse_max = max(rmse_max, rmse.max())
        if model_name[:-1] != next_model_name[:-1]:
            plot_data(rmses=rmses, plot_idx=plot_idx, model_name=model_name)
            rmses = []
            plot_idx += 1
    del performance_dict["xxx"]  # Remove dummy helper entry

    ax.plot([cfg.testing.teacher_forcing_steps, cfg.testing.teacher_forcing_steps], [0, rmse_max], ls="--",
            color="grey", label="End of teacher forcing")
    ax.grid()
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Time step")
    ax.set_xlim([1, len(rmse)])
    #ax.set_ylim([0.02, 0.35])
    #ax.set_ylim([0, 2200])
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig("rmse_plot.pdf")
    plt.close()

def plot_image_series(
    performance_dict: Dict,
    delta_step: int = 2,
    model_names: List = None,
    cmap: str = "viridis",
    sample: int = 0,
    plot_name: str = "image_series"
) -> None:
    """
    Create a plot of the ground truth image series in the top row, followed by each evaluated model in the rows below.
    
    :param performance_dict: Dict containing each evaluated model name as key and the according iot_dataset as value
    :param delta_step: Each delta_step-th image will be plotted of the time series
    :param model_names: A list of model names used as ylabels to indicate each row
    :param cmap: The colormap to visualize the data
    :param sample: Index of the sample to be visualized
    :param plot_name: The name or directory where to save the plot
    """
    #cmap = "Blues"
    def plot_single_image_series(axs, data: np.array, y_label: str, vmax: float, add_time_steps: bool = False):
        axs[0].set_ylabel(y_label)
        for ax_idx, ax in enumerate(axs):
            ax.set(frame_on=False, xticks=[], yticks=[])
            im = ax.imshow(data[ax_idx], origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
            if add_time_steps:
                title = fr"$t={ax_idx*delta_step}$" if ax_idx == 0 else ax_idx*delta_step
                ax.set_title(title)
        return axs, im

    key0 = list(performance_dict.keys())[0]
    n_rows = 1 + len(performance_dict.keys())  # Number of models plus ground truth
    n_cols = int(np.ceil(performance_dict[key0].sizes["time"]/delta_step))
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(0.75*n_cols, 0.8*n_rows), sharex=True, sharey=True)

    # Plot ground truth target image series in first row
    ds = performance_dict[key0]
    gt = ds.isel(sample=sample, dim_out=0).targets[::delta_step]
    vmax = max(gt.min().values, gt.max().values)
    axs[0], im = plot_single_image_series(axs=axs[0], data=gt.values, y_label="Target", vmax=vmax, add_time_steps=True)

    # Plot each model's output series in subsequent rows
    for m_idx, model_name in enumerate(performance_dict):
        data = performance_dict[model_name].isel(sample=sample, dim_out=0).outputs[::delta_step].values
        if model_names is not None and len(model_names) == n_rows-1: model_name = model_names[m_idx]
        plot_single_image_series(axs=axs[m_idx+1], data=data, y_label=model_name, vmax=vmax)
    
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=0.05, hspace=0.01)
    plt.subplots_adjust(wspace=0.0, hspace=0.2)
    #fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.0145, pad=0.01)
    #fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.03, pad=0.01)
    fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.035, pad=0.01)
    fig.savefig(f"{plot_name}.pdf", bbox_inches="tight")

def write_iot_dataset(
    cfg: DictConfig,
    inputs: np.array,
    outputs: np.array,
    targets: np.array,
    file_path: str
) -> None:
    """
    Creates a netCDF dataset for inputs, outputs, and targets and writes it to file.
    
    :param cfg: The hydra configuration of the model
    :param inits: The inputs to the model
    :param outputs: The outputs of the model (predictions)
    :param targets: The ground truth and target for prediction
    :param file_path: The path to the directory where the datasets are written to
    """
    if cfg.verbose: print("Building and writing input/output/target dataset")

    # Determine data dimensions
    D_in = inputs.shape[2]
    B, T, D_out, H, W = outputs.shape

    # Set up netCDF dataset
    coords = {}
    coords["sample"] = np.array(range(B), dtype=np.int32)
    coords["time"] = np.array(range(T), dtype=np.int32)
    coords["dim_in"] = np.array(range(D_in), dtype=np.int32)
    coords["dim_out"] = np.array(range(D_in), dtype=np.int32)
    coords["height"] = np.array(range(H), dtype=np.int32)
    coords["width"] = np.array(range(W), dtype=np.int32)
    chunkdict = {coord: len(coords[coord]) for coord in coords}
    chunkdict["sample"] = 1

    data_vars = {
        "inputs": (["sample", "time", "dim_in", "height", "width"], inputs),
        "outputs": (["sample", "time", "dim_out", "height", "width"], outputs),
        "targets": (["sample", "time", "dim_out", "height", "width"], targets),
    }

    ds = xr.Dataset(
        coords=coords,
        data_vars=data_vars,
    ).chunk(chunkdict)

    # Write dataset to file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path): os.remove(file_path)  # Delete file if it exists
    if cfg.verbose:
        write_job = ds.to_netcdf(file_path, compute=False)
        with ProgressBar(): write_job.compute()
        print()
    else:
        ds.to_netcdf(os.path.join(file_path)) 
