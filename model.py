import torch
import torch.nn as nn
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP
from types import SimpleNamespace

from utils import SafeSliceFunction


class UniversalHyperNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.main_net = MLP(
            n_in=args.largest_text_dim, hidden_layers=[],
            n_out=args.largest_image_dim, no_weights=True
        )
        self.hnet = HMLP(
            self.main_net.param_shapes, uncond_in_size=0,
            cond_in_size=args.hnet_cond_emb_dim, layers=[],
            num_cond_embs=args.num_image_encoders
        )
        self.hnet.apply_hyperfan_init(mnet=self.main_net)
        self.hnet = self.hnet.to(args.device)
        self.main_net = None
        self.return_params = args.return_params

    def eval(self):
        self.hnet.eval()

    def train(self):
        self.hnet.train()

    def forward(self, conditions, input_features, slice_dims):
        (D_img, D_txt) = slice_dims
        num_conds = 0

        if type(conditions) != list:
            num_conds = 1
        else:
            num_conds = len(conditions)

        params = self.hnet(cond_id=conditions)

        if self.return_params:
            return params

        # all items in params have the same shape, and the same encoder dim
        # is taken per step. So, just make the mask once
        # > mask for weight
        D1, D2 = self.args.largest_image_dim, self.args.largest_text_dim
        weight_mask = torch.zeros((D1, D2), requires_grad=False).to(input_features.device)
        weight_mask[:D_img, :D_txt] = 1

        # > mask for bias
        bias_mask = torch.zeros((D1,), requires_grad=False).to(input_features.device)
        bias_mask[:D_img] = 1

        output = []
        for i in range(num_conds):
            weight = params[i][0] * weight_mask
            bias = params[i][1] * bias_mask

            # if (D_img, D_txt) = (384, 768), then `mapped_features` has shape: [batch_size, args.largest_image_dim]
            # but only `mapped_features[:, :D_img]` will be non-zero
            mapped_features = input_features @ weight.T + bias
            output.append(mapped_features)

        return output


class Custom1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conditional_embeddings = nn.Embeddings(args.num_image_encoders, args.hnet_cond_emb_dim)
        self.to_weight = nn.Parameter(torch.empty(
            args.hnet_cond_emb_dim,
            args.largest_image_dim * args.largest_text_dim
        ))
        self.to_bias = nn.Parameter(torch.empty(args.hnet_cond_emb_dim, args.largest_image_dim))

    def init_params(self):
        pass

    def forward(self, x, D_img, D_txt):
        x = torch.tensor(x).long().to(self.args.device)
        embs = self.conditional_embeddings(x)
        for i in range(len(x)):
            weight = (embs[i] @ self.to_weight.T)[:D_img * D_txt]
            weight = weight.view(D_img, D_txt)
            bias = (embs[i] @ self.to_bias)[:D_img]


class Custom2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cond_embs = nn.Embedding(args.num_image_encoders, args.hnet_cond_emb_dim)
        # project to text dim
        self.fc1 = nn.Linear(args.hnet_cond_emb_dim, args.largest_text_dim)
        self.cn0 = nn.Conv1d(1, 256, 1, 1)
        # for all image dims we have
        self.cn1 = nn.Conv1d(256, 384, 1, 1)
        self.cn2 = nn.Conv1d(256, 768, 1, 1)
        self.cn3 = nn.Conv1d(256, 1024, 1, 1)
        self.act = nn.ReLU()

    def forward(self, conditions, input_features, slice_dims):
        (D_img, D_txt) = slice_dims
        num_conds = len(conditions) if type(conditions) == list else 1
        conditions = torch.tensor(conditions).to(input_features.device)
        cond_embs = self.cond_embs(conditions)

        outputs = []
        for i in range(num_conds):
            x = self.fc1(cond_embs[i]) # at shape: [D_txt]
            x = x.unsqueeze(0).unsqueeze(0) # at shape: [1, 1, D_txt]
            x = self.act(self.cn0(x))

            if D_img == 384:
                x = self.cn1(x)

            elif D_img == 768:
                x = self.cn2(x)

            elif D_img == 1024:
                x = self.cn3(x)

            mapped_features = input_features @ x.squeeze(0).T
            outputs.append(mapped_features)

        return outputs



def test():
    args = SimpleNamespace(**{})
    args.device = "cpu"
    args.largest_text_dim = 768
    args.largest_image_dim = 1024
    args.hnet_cond_emb_dim = 64
    args.num_image_encoders = 5

    model = UniversalHyperNetwork(args)

    input_features = torch.randn(16, 768)
    conditions = [0, 1, 2]
    slice_dims = (384, 768)
    output = model(conditions, input_features, slice_dims)

    for item in output:
        print(item.shape)


if __name__ == "__main__":
    test()
