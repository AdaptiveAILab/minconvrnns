import argparse
import os
import sys
from typing import List
import hydra
import numpy as np
import torch
import tqdm
import xarray as xr
from omegaconf import DictConfig

sys.path.append("")
from models import *
from data.pytorch_datasets import *
from utils.evaluation_tools import compute_metrics, generate_mp4, plot_rmse_over_time, plot_image_series, write_iot_dataset

def predict(
    cfg: DictConfig,
    dataset: torch.utils.data.DataLoader,
    model: torch.nn.Module
):
    """
    Generates predictions with the given model using the provided dataset. Optionally permutes inputs, if specified.

    :param cfg: The configuration of the model
    :param dataloader: The dataloader to prepare the data
    :param model: The ML model to generate predictions with
    """

    device = torch.device(cfg.device)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.testing.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Evaluate (without gradients): iterate over all test samples
    with torch.no_grad():
        inputs = list(); outputs = list(); targets = list()
        # Load data and generate predictions
        #for inpt, trgt in dataloader:
        for inpt, trgt in tqdm.tqdm(dataloader, desc=f"Generating predictions"):
            otpt = model(x=inpt.to(device=device), tf_steps=cfg.testing.teacher_forcing_steps)#, p_input_drop=0.0)
            inputs.append(inpt); outputs.append(otpt.cpu()); targets.append(trgt)
            break

        inputs, outputs, targets = torch.cat(inputs).numpy(), torch.cat(outputs).numpy(), torch.cat(targets).numpy()

    # Denormalize data if normalized for training
    if "normalize" in cfg.data.keys() and cfg.data.normalize:
        mean, std = dataset.stats["mean"], dataset.stats["std"]
        inputs = (inputs+mean)*std
        outputs = (outputs+mean)*std
        targets = (targets+mean)*std

    return inputs, outputs, targets


def evaluate_model(cfg: DictConfig) -> [np.array, np.array, np.array]:
    """
    Evaluates a single model for a given configuration.

    :param cfg: The hydra configuration for the model
    :return: Three arrays for input, outputs, targets, respectively. All of shape [B, T, C, H, W]
    """
    if cfg.verbose: print("\n\nInitialize dataloader and model")
    device = torch.device(cfg.device)

    # Initializing dataset for testing
    dataset = hydra.utils.instantiate(
        config=cfg.data,
        mode="test",
        sequence_length=cfg.testing.sequence_length
    )

    # Set up model
    model = hydra.utils.instantiate(config=cfg.model).to(device=device)
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters")

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_best.ckpt")
    #checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_last.ckpt")
    if cfg.verbose: print(f"\tRestoring model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Generate predictions
    if cfg.verbose: print("\nGenerate predictions")
    inputs, outputs, targets = predict(cfg=cfg, dataset=dataset, model=model)

    return inputs, outputs, targets


def run_evaluations(
    configuration_dir_list: str,
    device: str,
    overide: bool = False,
    batch_size: int = None,
    fpem_reps: int = 10,
    silent: bool = False,
    model_names: List = None,
    delta_step: int = 3
):
    """
    Evaluates a model with the given configuration.

    :param configuration_dir_list: A list of hydra configuration directories to the models for evaluation
    :param device: The device where the evaluations are performed
    """

    performance_dict = {}

    for configuration_dir in configuration_dir_list:
        # Initialize the hydra configurations for this model
        with hydra.initialize(version_base=None, config_path=os.path.join("..", configuration_dir, ".hydra")):
            cfg = hydra.compose(config_name="config")
            cfg.device = device
            if batch_size: cfg.testing.batch_size = batch_size

        if cfg.seed:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        cfg.testing.teacher_forcing_steps = 20
        cfg.testing.sequence_length = 96
        #cfg.testing.sequence_length = 30

        # Try to load input-output-target (iot) dataset if it exists, otherwise generate it
        src_path = os.path.join("outputs", str(cfg.model.name), "evaluation")
        file_path = os.path.join(src_path, "iot_dataset.nc")
        if not os.path.exists(file_path) or overide:
            inputs, outputs, targets = evaluate_model(cfg=cfg)
            write_iot_dataset(cfg=cfg, inputs=inputs, outputs=outputs, targets=targets, file_path=file_path)
        ds = xr.open_dataset(file_path)

        # Compute model prediction error metrics
        compute_metrics(cfg=cfg, ds=ds)

        # Generate video showcasing model prediction
        file_path = os.path.join(src_path, "video.mp4")
        if not os.path.exists(file_path) or overide:
            generate_mp4(cfg=cfg, ds=ds, file_path=file_path)

        performance_dict[cfg.model.name] = ds

    plot_rmse_over_time(cfg=cfg, performance_dict=performance_dict, model_names=model_names)
    plot_image_series(
        performance_dict=performance_dict, delta_step=delta_step, plot_name="image_series", model_names=model_names
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with a given configuration. Particular properties of the configuration can be "
                    "overwritten, as listed by the -h flag.")
    parser.add_argument("-c", "--configuration-dir-list", nargs='*', default=["configs"],
                        help="List of directories where the configuration files of all models to be evaluated lie.")
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="The device to run the evaluation. Any of ['cpu' (default), 'cuda', 'mpg'].")
    parser.add_argument("-o", "--overide", action="store_true",
                        help="Overide model predictions and evaluation files if they exist already.")
    parser.add_argument("-b", "--batch-size", type=int, default=None,
                        help="Batch size used for evaluation. Defaults to None, using test batch size from config.")
    parser.add_argument("-s", "--silent", action="store_true",
                        help="Silent mode to prevent printing results to console and visualizing plots dynamically.")
    parser.add_argument("-m", "--model-names", nargs='+', default=None, required=False,
                        help="A list of model names that are evaluated.")
    parser.add_argument("-ds", "--delta-step", type=int, default=3,
                        help="The skip rate between images for visualizing the image series.")

    run_args = parser.parse_args()
    run_evaluations(
        configuration_dir_list=run_args.configuration_dir_list,
        device=run_args.device,
        overide=run_args.overide,
        batch_size=run_args.batch_size,
        silent=run_args.silent,
        model_names=run_args.model_names,
        delta_step=run_args.delta_step
    )
    
    print("Done.")
