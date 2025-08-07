from typing import List
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

try:
    from helpers import CylinderPad
except:
    import sys
    sys.path.append(".")
    from utils.helpers import CylinderPad

class MinConvGRUCell(nn.Module):

    def __init__(
        self,
        batch_size: int = 1,
        input_size: int = 1,
        hidden_size: int = 16,
        height: int = 16,
        width: int = 16,
        device: torch.device = torch.device("cpu"),
        bias: bool = True,
        padding_mode: str = "zeros",
        **kwargs
    ):
        super().__init__()

        # Parameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.bias = bias
        self.device = device

        self.h = torch.zeros(size=(batch_size, hidden_size, height, width), device=device)

        # Convolution weights
        conv = []
        if padding_mode == "cylinder": conv.append(CylinderPad(padding=1))
        conv.append(nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size*2,
            kernel_size=3,
            stride=1,
            padding=0 if padding_mode == "cylinder" else 1,
            padding_mode="zeros" if padding_mode == "cylinder" else padding_mode,
            bias=bias
        ))
        self.conv = nn.Sequential(*conv)

    def reset_states(self, batch_size, height: int = 64, width: int = 64):
        if self.batch_size == batch_size and self.height == height and self.width == width:
            self.h = torch.zeros_like(self.h)
        else:
            self.batch_size = batch_size
            self.height = height
            self.width = width
            self.h = torch.zeros(size=(batch_size, self.hidden_size, height, width), device=self.device)

    @staticmethod
    def g(x):
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    @staticmethod
    def log_g(x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    @staticmethod
    def parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
        # log_coeffs: [B, T, hid, h, w]
        # log_values: [B, T+1, hid, h, w]

        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 0, 0, 0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star

        return torch.exp(log_h)[:, 1:].contiguous()  # [B, T, hid, h, w]

    def forward(
        self,
        x: torch.Tensor,
    ):
        # full sequence prediction in parallel #

        bt, c, h, w = x.shape
        b = self.batch_size
        t = int(bt/b)

        h0 = self.h.unsqueeze(1)

        k, h_tilde = torch.chunk(self.conv(x).view(b, t, self.hidden_size*2, h, w).contiguous(), chunks=2, dim=2)

        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h0 = self.log_g(h0)
        log_h_tilde = self.log_g(h_tilde)
        out = self.parallel_scan_log(log_coeffs, torch.cat([log_h0, log_z + log_h_tilde], dim=1))
        self.h = out[:, -1]
        out = out.view(bt, self.hidden_size, h, w)

        return out

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        # one time step prediction #

        h_prev = self.h if h_prev is None else h_prev

        z, h_tilde = torch.chunk(self.conv(x_t), chunks=2, dim=1)

        z = torch.sigmoid(z)
        h_tilde = self.g(h_tilde)
        h_curr = (1 - z) * h_prev + z * h_tilde

        self.h = h_curr

        return h_curr


class MinConvGRU(nn.Module):
    """
    A MinConvGRU implementation using Conv2d operations.
    """

    def __init__(
        self,
        batch_size: int = 8,
        input_size: int = 1,
        hidden_sizes: List = [16, 16],
        height: int = 16,
        width: int = 16,
        device: torch.device = torch.device("cpu"),
        bias: bool = True,
        padding_mode: str = "zeros",
        tanh_encoder: bool = False,
        norm = None,  # Must be overridden with a OmegaConfig
        **kwargs
    ):
        super().__init__()

        self.device = device

        self.encoder = []
        self.encoder.append(torch.nn.Conv2d(
            in_channels=input_size,
            out_channels=hidden_sizes[0],
            kernel_size=1  # When using padding, pass the padding_mode here and use CylinderPad with geopotential
        ))
        if tanh_encoder: self.encoder.append(torch.nn.Tanh())
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.cgru = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for h in hidden_sizes:
            self.cgru.append(MinConvGRUCell(
                batch_size=batch_size,
                input_size=h,
                hidden_size=h,
                height=height,
                width=width,
                device=device,
                bias=bias,
                padding_mode=padding_mode
            ))
            self.norms.append(hydra.utils.instantiate(norm))

        self.decoder = torch.nn.Conv2d(
            in_channels=hidden_sizes[-1],
            out_channels=input_size,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor, tf_steps: int = 10) -> torch.Tensor:
        b, t, c, h, w = x.shape
        tf_steps = min(tf_steps, t)
        self.reset(batch_size=b, height=h, width=w)

        x_tf = x[:, :tf_steps].reshape(b*tf_steps, c, h, w)
        outs = []

        # parallel mode for teacher forcing
        x_tf = self.encoder(x_tf)
        for cgru_cell, norm in zip(self.cgru, self.norms):
            # apply skip connection + post layer normalization
            z = x_tf
            z = cgru_cell(z)
            x_tf = z + x_tf
            x_tf = norm(x_tf)
        out = self.decoder(x_tf).view(b, tf_steps, c, h, w)
        outs.append(out)

        # sequential mode for closed loop prediction
        x_t = out[:, -1]
        # Iterate over sequence
        for t in range(t-tf_steps):
            # Forward the current time step's input through the model
            x_t = self.encoder(x_t)  # [B, C, H, W] C -> num_channels (hidden state)
            for cgru_cell, norm in zip(self.cgru, self.norms):
                z_t = x_t
                z_t = cgru_cell.step(z_t)  # [B, C, H, W]
                x_t = x_t + z_t  # residual connection
                x_t = norm(x_t)
            x_t = self.decoder(x_t) # [B, 1, I, H, W]
            outs.append(x_t.unsqueeze(1))

        outs = torch.cat(outs, dim=1)

        return outs

    def reset(self, batch_size: int = 8, height: int = 64, width: int = 64):
        for cgru_cell in self.cgru:
            cgru_cell.reset_states(batch_size=batch_size, height=height, width=width)


if __name__ == "__main__":
    # Example of how to build this model and feed it with data
    b, t, c, h, w = 8, 20, 1, 32, 32
    hidden_sizes = [4, 4]

    model = MinConvGRU(
        batch_size=b,
        input_size=c,
        hidden_sizes=hidden_sizes,
        height=h,
        width=w,
        norm=DictConfig({"_target_": "torch.nn.LayerNorm", "normalized_shape": [h, w]}),
        padding_mode="cylinder"
    )
    print(model)

    x = torch.randn(b, t, c, h, w)
    print("Input shape: ", x.shape)
    y_hat = model(x)
    print("Output shape:", y_hat.shape)
    print("Done.")
