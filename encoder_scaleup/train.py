# libraries
import math
from model import UniversalHyperNetwork
import torch
from torch.utils.data import DataLoader
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP
import numpy as np
import argparse
from contextlib import suppress
import warnings
from tqdm import tqdm

# local code imports
from data import *
from utils import *
from model import *

# shut off warnings
warnings.simplefilter("ignore")
# switch on anomaly detection
torch.autograd.set_detect_anomaly(True)


def train_one_epoch(args, hnet, data_objects, criterion, optimizer, scheduler, scaler, epoch):
    hnet.train()
    (dataset, indices_loader, encoder_loader) = data_objects

    # trackers
    accuracies = []
    corrects = {}
    total = 0
    running_loss = 0
    total_loss = 0
    autocast = suppress # NOTE: edit later
    logit_scale = torch.tensor(np.log(100.0)).to(args.device)

    num_steps = math.ceil(len(dataset) / args.batch_size)
    bar = tqdm(total=num_steps)
    for idx, batch_indices in enumerate(indices_loader):
        # first get the encoders for which we are observing the data stream
        encoder_indices, encoder_dims = next(encoder_loader)
        # get the data stream
        image_features, text_features = dataset.get_minibatch(batch_indices, encoder_indices, encoder_dims)
        [B, N, D_img] = image_features.shape
        # image_features = pad_image_features(image_features, args.largest_image_dim - D_img)
        D_txt = text_features.shape[-1]

        # cast embeddings
        image_features = image_features.float().to(args.device)
        text_features = text_features.float().to(args.device)

        # schedule if we want to
        if scheduler is not None:
            step = epoch * len(indices_loader) + idx
            scheduler(step)

        optimizer.zero_grad()

        outputs = None

        # forward pass
        params = []
        with autocast():
            params = hnet(cond_id=[i for i in range(N)])

            for i in range(N):
                weight = params[i][0][:D_img, :D_txt]
                bias = params[i][1][:D_img]
                mapped_text_features = text_features @ weight.T + bias
                mapped_text_features = mapped_text_features / mapped_text_features.norm(dim=-1, keepdim=True)

                loss, inbatch_corrects = criterion.compute_loss_and_accuracy(
                    logit_scale, image_features[:, i, :].view(B, D_img), mapped_text_features
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
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward(retain_graph=args.retain_graph)
            optimizer.step()

        bar.update(1)

    return {"avg_loss": running_loss / num_steps, "accuracies": accuracies}


def main(args):
    # load in training data
    kwargs = {
        "image_data_path": "../datasets/scaleup/random_image_embeddings.pt",
        "text_data_path": "../datasets/scaleup/random_text_embeddings.pt"
    }
    dataset = UniversalEmbeddings(**kwargs)
    index_loader = init_indices_loader(args, dataset)
    encoder_loader = init_encoder_loader(args, dataset)

    # load in the hyper-network
    args.num_image_encoders = dataset.num_image_encoders
    model = HMLP(
        [[args.largest_image_dim, args.largest_text_dim], [args.largest_image_dim]], uncond_in_size=0,
        cond_in_size=args.hnet_cond_emb_dim, layers=[],
        num_cond_embs=args.num_image_encoders
    )
    # optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None

    # loss function and grad-scaler
    criterion = ClipLoss()
    scaler = None

    # training loop
    for epoch in range(args.num_epochs):
        train_logs = train_one_epoch(
            args, model, (dataset, index_loader, encoder_loader),
            criterion, optimizer, scheduler, scaler, epoch
        )
        print(train_logs)
        break



def setup_args():
    parser = argparse.ArgumentParser()
    # overall settings
    parser.add_argument("--device", type=str, default="cpu")
    # model settings
    parser.add_argument("--largest-image-dim", type=int, default=1024)
    parser.add_argument("--largest-text-dim", type=int, default=768)
    parser.add_argument("--num-image-encoders", type=int, default=1)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--hnet-cond-emb-dim", type=int, default=64)
    parser.add_argument("--return-params", type=bool, default=False)
    # training settings
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--retain-graph", type=bool, default=False)

    # return args
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args()
    main(args)
