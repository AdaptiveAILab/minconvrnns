import glob
import os
import time
from typing import List
import numpy as np
import pandas as pd
import torch
import xarray as xr

class GeopotentialDataset(torch.utils.data.Dataset):

    # Statistics are computed over the training period from 1979-01-01 to 2014-12-31
    STATISTICS = {
        "mean": 54107.863,
        "std": 3349.0322
    }

    def __init__(
            self,
            data_path: List = ["data", "netcdf", "geopotential_500_5.625deg"],
            train_start_date: np.datetime64 = np.datetime64("1979-01-01"),
            train_end_date: np.datetime64 = np.datetime64("2014-12-31"),
            val_start_date: np.datetime64 = np.datetime64("2015-01-01"),
            val_end_date: np.datetime64 = np.datetime64("2016-12-31"),
            test_start_date: np.datetime64 = np.datetime64("2017-01-01"),
            test_end_date: np.datetime64 = np.datetime64("2018-12-31"),
            timedelta: int = 1,
            init_dates: np.array = None,
            sequence_length: int = 24,
            noise: float = 0.0,
            normalize: bool = False,
            downscale_factor: int = 1,
            context_size: int = 1,
            engine: str = "netcdf4",
            height: int = 32,
            width: int = 64,
            mode: str = "train",
            **kwargs
        ):
        """
        Constructor of a pytorch dataset module.

        :param data_start_date: The first time step of the data on the disk
        :param data_stop_date: The last time step of the data on the disk
        :param data_src_path: The source path of the data
        :param sequence_length: The number of time steps used for training
        """

        self.stats = GeopotentialDataset.STATISTICS
        self.timedelta = timedelta
        self.sequence_length = sequence_length
        self.noise = noise
        self.normalize = normalize
        self.downscale_factor = downscale_factor
        self.context_size = context_size
        self.init_dates = init_dates
        self.p = "z"

        if mode == "train":
            start_date = train_start_date
            end_date = train_end_date
        elif mode == "val":
            start_date = val_start_date
            end_date = val_end_date
        elif mode == "test":
            start_date = test_start_date
            end_date = test_end_date
            self.init_dates = self._make_biweekly_inits(start_date=start_date, end_date=end_date)
        else:
            raise ValueError(f"Mode must be any of 'train', 'val', or 'test' but is '{mode}'.")

        # Get paths to all (yearly) netcdf/zarr files
        fpaths = glob.glob(os.path.join(*data_path, "*"))

        print(f"\tLoading dataset from {start_date} to {end_date} into RAM...", sep=" ", end=" ", flush=True)
        a = time.time()
        # Load the data as xarray dataset
        self.ds = xr.open_mfdataset(fpaths, engine=engine).sel(time=slice(start_date, end_date, timedelta))

        # Chunk and load dataset to memory
        chunkdict = dict(time=self.sequence_length+1, lat=height, lon=width)
        self.ds = self.ds.chunk(chunkdict).load()
        print(f"took {time.time()-a} seconds")

        # Downscale dataset if desired
        if downscale_factor > 1: self.ds = self.ds.coarsen(lat=downscale_factor, lon=downscale_factor).mean()

    def __len__(self):
        if self.init_dates is None:
            # Randomly sample initialization dates from the dataset
            return (self.ds.sizes["time"]-self.sequence_length)//self.sequence_length
        else:
            return len(self.init_dates)

    def __getitem__(self, item):
        """
        return: Two tensors for inputs and targets, both of shape [batch, time, dim, height, width].
        """
        item = item*self.sequence_length if self.init_dates is None else item

        # Load the (normalized) prognostic variables of shape [time, #prognostic_vars, lat, lon] into memory
        if self.init_dates is None:
            lazy_data = self.ds[self.p].isel(time=slice(item, item+self.sequence_length+1))
        else:
            lazy_data = self.ds[self.p].sel(
                time=slice(self.init_dates[item],
                           self.init_dates[item]+pd.Timedelta(f"{(self.sequence_length+1)*self.timedelta}h"))
            )
        # Load the data to memory
        if self.normalize: lazy_data = (lazy_data-self.stats["mean"])/self.stats["std"]
        data = np.expand_dims(np.float32(lazy_data.compute()), axis=1)

        # Separate inputs and targets
        inpt = data[:-1] + np.float32(np.random.randn(*data[:-1].shape)*self.noise)
        trgt = data[1:]

        return inpt, trgt

    @staticmethod
    def _make_biweekly_inits(
        start_date: np.datetime64,
        end_date: np.datetime64,
        sequence_length: int = 168,
        timedelta: int = 1
    ) -> np.array:
        """
        Follows ECMWF's convention and builds biweekly inits to start two forecasts per week over a given time window.
        This is typically used for model evaluation.

        :param start_date: The starting date of the biweekly forecasts
        :param end_date: The last date of the biweekly forecasts
        :param sequence_length: The length of each forecast in time steps
        :param timedelta: The step size in hours
        :return: An array holding the initialization times
        """
        times1 = pd.date_range(start=start_date,
                               end=pd.Timestamp(end_date)-pd.Timedelta(hours=sequence_length*timedelta),
                               freq='7D')
        times2 = pd.date_range(start=pd.Timestamp(start_date)+pd.Timedelta(days=3),
                               end=pd.Timestamp(end_date)-pd.Timedelta(hours=sequence_length*timedelta),
                               freq='7D')
        return times1.append(times2).sort_values().to_numpy()


if __name__ == "__main__":

    # Example of creating a WeatherBench dataset for the PyTorch DataLoader
    dataset = GeopotentialDataset(
        data_path=["data", "netcdf", "geopotential_500_5.625deg"],
        train_end_date="1982-12-31",
        sequence_length=24,
        noise=0.0,
        normalize=True,
        downscale_factor=1
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    for inpts, trgts in train_dataloader:
        print(inpts.shape, trgts.shape)
        break

    print(dataset)