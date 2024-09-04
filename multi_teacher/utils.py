import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import *
import timm
import numpy as np


class ImageEncoder():
	def __init__(self, model_name, device="cuda"):
		self.model_name = model_name
		self.device = device

		self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
		self.model = self.model.to(self.device)
		self.model.eval()

		self.config = timm.data.resolve_model_data_config(self.model)
		self.transform = timm.data.create_transform(**self.config, is_training=False)

	def encode_image(self, image):
		try:
			x = image.shape
		except:
			image = self.transform(image)

		image_features = self.model(image)
		return F.normalize(image_features, dim=-1)


class MlpMapper(nn.Module):
	def __init__(self, input_dim: int, intermediate_dims: List, output_dim: int, use_bias: bool = True, logit_scale: float = 100.0):
		super().__init__()
		self.input_dim = input_dim
		self.intermediate_dims = intermediate_dims # list of ints
		self.output_dim = output_dim
		self.num_layers = len(intermediate_dims) + 1

		self.layers = []
		current_dim = input_dim
		next_dims = intermediate_dims + [output_dim]

		if logit_scale < 0:
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		else:
			self.logit_scale = torch.tensor(np.log(logit_scale))

		for i in range(self.num_layers):
			self.layers.append(nn.Linear(current_dim, next_dims[i], bias=use_bias))
			current_dim = next_dims[i]

			if i != self.num_layers - 1:
				self.layers.append(nn.GELU())

		self.layers = nn.Sequential(*self.layers)

	def forward(self, x):
		x = self.layers(x)
		return F.normalize(x, dim=-1)
