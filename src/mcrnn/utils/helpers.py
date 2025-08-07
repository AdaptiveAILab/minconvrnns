import os
import torch

class CylinderPad(torch.nn.Module):
	"""
	Implements circular padding along the longitude (horizontal) dimension and constant padding along the latitude
	(vertical) dimension to represent a cylinder.
	"""
	def __init__(
    	self,
		padding: int = 1
	):
		super().__init__()
		self.p = padding
		
	def forward(self, x):
		x = torch.nn.functional.pad(input=x, pad=(self.p, self.p, 0, 0), mode="circular")  # Longitude (horizonal)
		x = torch.nn.functional.pad(input=x, pad=(0, 0, self.p, self.p), mode="constant")  # Latitude (vertical)
		return x

def write_checkpoint(
		model,
		optimizer,
		scheduler,
		epoch: int,
		iteration: int,
		best_val_error: float,
		dst_path: str
	):
	"""
	Writes a checkpoint including model, optimizer, and scheduler state dictionaries along with current epoch,
	iteration, and best validation error to file.
	
	:param model: The network model
	:param optimizer: The pytorch optimizer
	:param scheduler: The pytorch learning rate scheduler
	:param epoch: Current training epoch
	:param iteration: Current training iteration
	:param best_val_error: The best validation error of the current training
	:param dst_path: Path where the checkpoint is written to
	"""
	os.makedirs(os.path.dirname(dst_path), exist_ok=True)
	torch.save(obj={"model_state_dict": model.state_dict(),
				 "optimizer_state_dict": optimizer.state_dict(),
				 "scheduler_state_dict": scheduler if scheduler is None else scheduler.state_dict(),
				 "epoch": epoch + 1,
				 "iteration": iteration,
				 "best_val_error": best_val_error},
			f=dst_path)
