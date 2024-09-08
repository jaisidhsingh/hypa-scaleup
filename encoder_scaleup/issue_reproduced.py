import torch
import torch.nn as nn
import torch.nn.functional as F
from hypnettorch.hnets import HMLP
import numpy as np
import argparse


def clip_loss(logit_scale, image_features, text_features):
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, dtype=torch.long).to(image_features.device)

    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T

    loss = F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
    return loss / 2


def train_attempt(args):
    """
    -----
    Goal:
    ----- 
    Train a hyper-network which outputs the largest possible mapping, i.e.,
    (768 x 768). Encoder pairs which have smaller embed dims will be mapped by a slice
    of this largest possible mapping, for e.g., if we want to map from 768 to 384, then
    mapping = lambda x: x @ predicted_weight[:384, :].T + predicted_bias[:384]

    ------------------------------------
    Implementation details of this test:
    ------------------------------------
    Let us say we have 6 image encoders: (i)  3 of embed dim = 384, 
    (ii) 3 of embed dim = 768. Also, we have only 2 steps per epoch. 
    In the first step, we sample the image encoders of embed dim = 384. 
    The next step uses image encoders of embed dim = 768. 
    """

    # load the hyper-network
    mapper_D_in = 64 # fixed text encoder
    mapper_D_out_over_steps = [32, 16]
    largest_image_embed_dim = max(mapper_D_out_over_steps)
    largest_mapping_shape = [ [largest_image_embed_dim, mapper_D_in], [largest_image_embed_dim] ]

    model = HMLP(largest_mapping_shape, layers=[], cond_in_size=8, num_cond_embs=4, uncond_in_size=0)

    active_image_encoders_over_steps = [
        [jk for jk in range(2, 4)],
        [ij for ij in range(0, 2)],
    ]

    # dataset of embeddings
    batch_size = 4
    dataset = [
        # second batch of the dataset
        (
            torch.randn(batch_size, 2, 32), # image_features
            torch.randn(batch_size, mapper_D_in) # text_features
        ),
        # first batch of the dataset
        (
            torch.randn(batch_size, 2, 16), # image_features
            torch.randn(batch_size, mapper_D_in) # text_features
        ),
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 2
    num_batches = len(dataset)
    logit_scale = torch.tensor(np.log(100.0))

    for epoch in range(num_epochs):
        # train loop
        total_loss = 0

        for idx in range(num_batches):
            step = epoch * (num_batches) + idx
            
            (image_features, text_features) = dataset[idx]
            N = image_features.shape[1]

            if args.solve_with_gradacc:
                if idx == 0:
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()

            cond_ids = active_image_encoders_over_steps[idx]
            params = model(cond_id=cond_ids)

            D_out = mapper_D_out_over_steps[idx]

            for j in range(N):
                mapped_text_features = text_features @ params[j][0][:D_out, :].T + params[j][1][:D_out]
                loss = clip_loss(logit_scale, image_features[:, j, :], mapped_text_features)
                total_loss += loss
            
            total_loss = total_loss / N

            try:
                total_loss.backward(retain_graph=args.retain_graph)
                
                if args.solve_with_gradacc:
                    if idx == num_batches-1:
                        optimizer.step()
                else:
                    optimizer.step()

                print(f"Training step {step+1} (epoch {epoch+1}) completed")
                print(f"Image encoder embed dim here is {D_out}")
                print(" ")
            
            except Exception as error:
                print("Error occured at step", step+1, f"(epoch {epoch+1})")
                print(f"Image encoder embed dim here is {D_out}")
                print("----------------------------")
                print(error)
                print(" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retain-graph", type=bool, default=False)
    parser.add_argument("--solve-with-gradacc", type=bool, default=False)
    args = parser.parse_args()
    train_attempt(args)
