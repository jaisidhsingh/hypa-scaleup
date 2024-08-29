import torch
from torch.utils.data import DataLoader
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP
import numpy as np

from data import *


def train_one_epoch(args, models, data_objects, criterion, optimizer, scheduler, scaler, epoch):
    (hnet, main_net) = models
    hnet.train()
    (dataset, indices_loader, encoder_loader) = data_objects

    # trackers
    accuracies = []
    corrects = {}
    total = 0
    running_loss = 0
    total_loss = 0
    autocast = torch.cuda.amp.autocast
    logit_scale = torch.tensor(np.log(100.0)).to(args.device)

    for idx, batch_indices in enumerate(indices_loader):
        # first get the encoders for which we are observing the data stream
        encoder_indices, encoder_dims = next(encoder_loader)
        # get the data stream
        image_features, text_features = dataset.get_minibatch(batch_indices, encoder_indices, encoder_dims)
        [B, N, D] = image_features.shape

        # cast embeddings
        image_features = image_features.float().to(args.device)
        text_features = text_features.float().to(args.device)

        # schedule if we want to
        if scheduler is not None:
            step = epoch * len(indices_loader) + idx
            scheduler(step)

        optimizer.zero_grad()

        # forward pass
        with autocast():
            params = hnet(cond_id=encoder_indices)
            for i in range(N):
                mapped_text_features = main_net(text_features, weights=params[i])
                loss, inbatch_corrects = criterion.compute_loss_and_accuracy(
                    logit_scale, image_features[:, i, :].view(B, D), mapped_text_features
                )

                total_loss += loss
                running_loss += loss.item()

                if i not in corrects:
                    corrects[i] = 0
                corrects[i] += inbatch_corrects

        total += B
        accuracies = [round(val/total * 100, 2) for val in corrects.values()]
        total_loss = total_loss / N
        running_loss /= N

        # backward pass
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return {"avg_loss": running_loss / len(indices_loader), "accuracies": accuracies}


def main(args):
    kwargs = {}
    dataset = UniversalEmbeddings(kwargs)
    index_loader = init_indices_loader(args, dataset)
    encoder_loader = init_encoder_loader(args, dataset)

    main_net = MLP(n_in=args.largest_text_dim, hidden_layers=[], n_out=args.largest_image_dim).to(args.device)
    hnet = HMLP(main_net.param_shapes, uncond_emb_size=0, cond_emb_size=args.image_embed_dim, num_cond_embs=args.num_image_encoders)
    hnet.apply_hyperfan_init(mnet=main_net)
    hnet = hnet.to(args.device)
