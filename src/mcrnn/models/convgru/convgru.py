from typing import List
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

try:
    from helpers import CylinderPad
except:
    import sys
    sys.path.append(".")
    from utils.helpers import CylinderPad

class ConvGRUCell(nn.Module):

    def __init__(
        self,
        batch_size,
        input_size,
        hidden_size,
        height,
        width,
        device,
        bias=True,
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

        # Hidden state
        self.h = torch.zeros(size=(batch_size, hidden_size, height, width), device=device)

        # Convolution weights
        conv_a = []
        conv_b = []
        if padding_mode == "cylinder":
            conv_a.append(CylinderPad(padding=1))
            conv_b.append(CylinderPad(padding=1))
        conv_a.append(nn.Conv2d(
            in_channels=input_size + hidden_size,
            out_channels=hidden_size*2,
            kernel_size=3,
            stride=1,
            padding=0 if padding_mode == "cylinder" else 1,
            padding_mode="zeros" if padding_mode == "cylinder" else padding_mode,
            bias=bias
        ))
        conv_b.append(nn.Conv2d(
            in_channels=input_size + hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            padding=0 if padding_mode == "cylinder" else 1,
            padding_mode="zeros" if padding_mode == "cylinder" else padding_mode,
            bias=bias
        ))
        self.conv_a = nn.Sequential(*conv_a)
        self.conv_b = nn.Sequential(*conv_b)

    def reset_states(self, batch_size, height: int = 64, width: int = 64):
        if self.batch_size == batch_size and self.height == height and self.width == width:
            self.h = torch.zeros_like(self.h)
        else:
            self.batch_size = batch_size
            self.height = height
            self.width = width
            self.h = torch.zeros(size=(batch_size, self.hidden_size, height, width), device=self.device)

    def reset_parameters(self):
        # Uniform distribution initialization of GRU weights with respect to
        # the number of GRU cells in the layer
        std = 1.0/np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor = None,
    ):

        h_prev = self.h if h_prev is None else h_prev

        # Compute update and reset gates and candidate cell state
        z, r = torch.chunk(torch.sigmoid(self.conv_a(torch.cat([x, h_prev], dim=1))), chunks=2, dim=1)
        h_cand = torch.tanh(self.conv_b(torch.cat([x,r*h_prev], dim=1)))

        # Update hidden state
        h_curr = (1-z)*h_prev + z*h_cand
        self.h = h_curr

        return h_curr


class ConvGRU(nn.Module):
    """
    A ConvGRU implementation using Conv2d operations.
    """

    def __init__(
        self,
        batch_size: int = 8,
        input_size: int = 1,
        hidden_sizes: List = [4, 4],
        height: int = 16,
        width: int = 16,
        device: torch.device = torch.device("cpu"),
        bias: bool = True,
        padding_mode: str = "zeros",
        tanh_encoder: bool = False,
        norm = None,
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

        self.cgru= torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for h in hidden_sizes:
            self.cgru.append(ConvGRUCell(
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
        self.cgru = torch.nn.ModuleList(self.cgru)

        self.decoder = torch.nn.Conv2d(
            in_channels=hidden_sizes[-1],
            out_channels=input_size,
            kernel_size=1
        )

    def forward(
        self,
        x: torch.Tensor,
        tf_steps: int = 10,  # teacher forcing steps
    ) -> torch.Tensor:

        # Zero initialize hidden and cell states of GRU
        b, t, c, h, w = x.shape
        self.reset(batch_size=b, height=h, width=w)
        outs = []

        # Iterate over sequence
        for t in range(x.shape[1]):
            # During teacher forcing, take the ground truth as input. In closed loop, take the last model output.
            x_t = x[:, t] if t < tf_steps else x_t
            # Forward the current time step's input through the model
            x_t = self.encoder(x_t)  # [b, hidden, h, w]
            for cgru_cell, norm in zip(self.cgru, self.norms):
                z_t = x_t
                z_t = cgru_cell(z_t)  # [b, hidden, h, w]
                x_t = x_t + z_t  # residual connection
                x_t = norm(x_t)
            x_t = self.decoder(z_t)
            outs.append(x_t)

        return torch.stack(outs, dim=1)

    def reset(self, batch_size: int = 8, height: int = 64, width: int = 64):
        for cgru_cell in self.cgru:
            cgru_cell.reset_states(batch_size=batch_size, height=height, width=width)
        self.zeros = torch.zeros(size=(batch_size, 1, height, width), device=self.device)


if __name__ == "__main__":
    # Example of how to build this model and feed it with data
    b, t, c, h, w = 8, 20, 1, 32, 32
    hidden_sizes = [4, 4]

    model = ConvGRU(
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
