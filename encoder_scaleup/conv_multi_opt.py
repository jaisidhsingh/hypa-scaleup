import sys
import torch
import numpy as np
import torch.nn.functional as F
import argparse
from model import ConvHyperNet
from issue_reproduced import clip_loss


def prepare_optimizers(model):
    decoder_params = {i: [] for i in model.dims["image"]}
    reversed_decoder_lookup = {v:k for k, v in model.decoder_lookup.items()}
    base_params = []
    
    for name, param in model.named_parameters():
        if "_decoder" in name:
            key = reversed_decoder_lookup[int(name.split(".")[1])]
            decoder_params[key].append(param)
        else:
            base_params.append(param)
    
    optimizers = {str(k): torch.optim.Adam(decoder_params[k], lr=1e-3) for k in decoder_params.keys()}
    optimizers.update({"base": torch.optim.Adam(base_params, lr=1e-3)})
    return optimizers


def train_attempt(args):
    # load the hyper-network
    mapper_D_in = 768 # fixed text encoder
    mapper_D_out_over_steps = [384, 768]

    dims = {"image": [384, 768], "text": 768, "num_embed_dims": 2, "intermediate": 256}
    model = ConvHyperNet(dims, num_cond_embs=6, cond_emb_dim=8)
    optimizers = prepare_optimizers(model)

    active_image_encoders_over_steps = [
        # 384 embed dim encoders are used first
        [0, 1, 2],
        # 768 embed dim encoders are used second
        [3, 4, 5]
    ]

    # dataset of embeddings
    batch_size = 4
    dataset = [
        # first batch of the dataset
        (
            torch.randn(batch_size, 3, 384), # image_features
            torch.randn(batch_size, mapper_D_in) # text_features
        ),
        # second batch of the dataset
        (
            torch.randn(batch_size, 3, 768), # image_features
            torch.randn(batch_size, mapper_D_in) # text_features
        )
    ]

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

            # zero grad
            if idx == 0:
                for k in optimizers.keys():
                    optimizers[k].zero_grad()

            D_out = mapper_D_out_over_steps[idx]
            cond_ids = active_image_encoders_over_steps[idx]
            params = model(cond_id=cond_ids, D_img=D_out)

            for j in range(N):
                mapped_text_features = text_features @ params[j][0][:D_out, :].T + params[j][1][:D_out]
                loss = clip_loss(logit_scale, image_features[:, j, :], mapped_text_features)
                total_loss += loss
            
            total_loss = total_loss / N

            try:
                total_loss.backward(retain_graph=args.retain_graph)

                if idx == 1:
                    optimizers["base"].step()
                    optimizers[str(D_out)].step()

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
    args = parser.parse_args()
    train_attempt(args)

