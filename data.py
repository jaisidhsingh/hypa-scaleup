import os
import torch
from torch.utils.data import Dataset
import random


class UniversalEmbeddings(Dataset):
    def __init__(self, image_data_path, text_data_path):
        self.image_embeddings = torch.load(image_data_path)
        self.image_encoder_data = {k: list(v.keys()) for k, v in self.image_embeddings.items()}
        self.image_encoders = [name for v in self.image_encoder_data.values() for name in v]
        self.num_image_encoders = sum([len(v) for v in self.image_encoder_data.values()])

        self.text_embeddings = torch.load(text_data_path)
        self.text_embed_dim = list(self.text_embeddings.keys())[0]
        self.text_encoder = list(self.text_embeddings[self.text_embed_dim].keys())[0]
        self.num_text_encoders = 1

        self.num_samples = self.text_embeddings[self.text_embed_dim][self.text_encoder].shape[0]


    def __len__(self):
        return self.num_samples

    def get_minibatch(self, batch_indices, sampled_encoder_indices, encoder_dims):
        # first get the encoder names from indices
        sampled_encoders = [self.image_encoders[i] for i in sampled_encoder_indices]
        print(sampled_encoders, encoder_dims[0])

        # remember to not shuffle the data
        start, end = batch_indices[0], batch_indices[-1]+1

        # now get the image embeddings
        chosen_dim = encoder_dims[0]
        image_embeddings = torch.cat(
            [self.image_embeddings[chosen_dim][k][start : end, :].unsqueeze(1) for k in sampled_encoders],
            dim=1
        )
        assert image_embeddings.shape == torch.Size([end-start, len(sampled_encoders), chosen_dim]), "Image embeddings are incorrectly stacked!"

        # get text embeddings
        text_embeddings = self.text_embeddings[self.text_embed_dim][self.text_encoder][start : end, :]
        assert text_embeddings.shape == torch.Size([end-start, self.text_embed_dim]), "Text embeddings are incorrectly stacked!"

        # return the data
        return image_embeddings, text_embeddings


def init_indices_loader(args, dataset):
    num_samples = len(dataset)
    batch_size = args.batch_size
    total_dataset_indices = [x for x in range(num_samples)]

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = total_dataset_indices[start_idx : end_idx]
        yield batch_indices


def init_encoder_loader(args, dataset):
    num_encoders = sum([len(v) for v in dataset.image_encoder_data.values()])
    batch_size = args.encoder_batch_size

    assert batch_size <= min([len(v) for v in dataset.image_encoder_data.values()])
    total_encoder_indices = [x for x in range(num_encoders)]

    # get info about the embedding dim of the sampled encoders too
    total_dim_store = [[k for _ in range(len(v))] for k, v in dataset.image_encoder_data.items()]
    total_dim_store = [y for x in total_dim_store for y in x]
    # check if all the dims are the same (mandatory)

    for start_idx in range(0, num_encoders, batch_size):
        end_idx = min(start_idx + batch_size, num_encoders)
        encoder_indices = total_encoder_indices[start_idx : end_idx]

        encoder_dims = total_dim_store[start_idx : end_idx]
        assert set(encoder_dims) == {encoder_dims[0]}, "All encoders in the batch are not of the same dimension => future error while concating!"

        yield encoder_indices, encoder_dims
