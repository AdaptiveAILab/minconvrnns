import os
from typing import List
import numpy as np
import torch
import xarray as xr

class NavierStokesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path: List = ["data", "netcdf", "navier-stokes"],
            train_set_name: str = "ns_r1e+03_n5_t50_s64.nc",
            val_set_name: str = "ns_r1e+03_n5_t50_s64.nc",
            test_set_name: str = "ns_r1e+03_n5_t50_s64.nc",
            mode: str = "train",
            noise: float = 0.0,
            sequence_length: int = 15,
            normalize: bool = False,
            downscale_factor: int = None,
            **kwargs
        ):
        """
        Constructor of a pytorch dataset module.
        """
        if mode == "train": dataset_name = train_set_name
        if mode == "val": dataset_name = val_set_name
        if mode == "test": dataset_name = test_set_name
        data_path = os.path.join(*data_path, dataset_name)

        self.sequence_length = sequence_length + 1
        self.noise = noise
        self.normalize = normalize
        self.downscale_factor = downscale_factor

        self.ds = xr.open_dataset(data_path)
        self.mean = self.ds.u.mean().values
        self.std = self.ds.u.std().values

        # Downscale dataset if desired
        if downscale_factor: self.ds = self.ds.coarsen(height=downscale_factor, width=downscale_factor).mean()
        
    def __len__(self):
        return self.ds.sizes["sample"]

    def __getitem__(self, item):
        r = np.random.randint(0, self.ds.sizes["time"]-self.sequence_length+1)
        x = np.float32(self.ds.u.isel(sample=item, time=slice(r, r+self.sequence_length-1)))
        x = x + np.float32(np.random.randn(*x.shape)*self.noise)
        y = np.float32(self.ds.u.isel(sample=item, time=slice(1+r, r+self.sequence_length)))
        return x, y
