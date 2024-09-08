import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import *


class ClipLoss():
    def compute_loss_and_accuracy(self, logit_scale, image_features, text_features):
        # get device
        device = image_features.device

        # remove the log from the logit scale
        logit_scale = logit_scale.exp().to(device)

        # get labels
        labels = torch.arange(image_features.shape[0], dtype=torch.long).to(device)

        # get similarity matrices
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # get loss
        loss = F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        loss = loss / 2

        # get the predictions that are correct
        preds = logits_per_image.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        return loss, correct


def pad_image_features(image_features, pad_length, multi_encoder=True):
    if multi_encoder:
        assert len(image_features.shape) == 3, "Expected embeddings of multiple encoders, and shape: [batch, num_encoders, dim]. Did not receive as expected!"
        [B, N, D] = image_features.shape
        pad = torch.zeros(B, N, pad_length).to(image_features.device)
        return torch.cat([image_features, pad], dim=2)

    else:
        assert len(image_features.shape) == 2, "Expected embeddings of one encoder, and shape: [batch, dim]. Did not receive as expected!"
        [B, D] = image_features.shape
        pad = torch.zeros(B, pad_length).to(image_features.device)
        return torch.cat([image_features, pad], dim=1)


class SafeSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, start_row, start_col, end_row, end_col):
        ctx.save_for_backward(torch.tensor([start_row, start_col, end_row, end_col]))
        ctx.input_shape = input_tensor.shape
        return input_tensor[start_row:end_row, start_col:end_col].clone()

    @staticmethod
    def backward(ctx, grad_output):
        start_row, start_col, end_row, end_col = ctx.saved_tensors[0]
        input_shape = ctx.input_shape
        grad_input = torch.zeros(input_shape, device=grad_output.device)
        grad_input[start_row:end_row, start_col:end_col] = grad_output
        return grad_input, None, None, None, None


def sliced_mapping(x, params, image_embed_dim, text_embed_dim):
    # the first item in `params` is the weight, second item is the bias
    return x @ params[0][:image_embed_dim, :text_embed_dim].T + params[1][:image_embed_dim]


def slice_weight_with_hook(slice_dims, full_dims):
    def hook(grad):
        mask = torch.zeros(full_dims).to(grad.device)
        mask[:slice_dims[0], :slice_dims[1]] = 1
        return grad * mask
    return hook


def slice_bias_with_hook(slice_dim, full_dim):
    def hook(grad):
        mask = torch.zeros(full_dim).to(grad.device)
        mask[:slice_dim] = 1
        return grad * mask
    return hook


def zero_out_sliced_grads(params, image_embed_dim, text_embed_dim):
    params[0].grad.zero_()
    params[1].grad.zero_()


def retain_grad_for_params(params):
    params[0].retain_grad()
    params[1].retain_grad()
