import torch
import torch.nn as nn
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP
from types import SimpleNamespace

from utils import SafeSliceFunction


class HyperNetwork(nn.Module):
    def __init__(self, param_shapes, cond_emb_dim, num_cond_embs):
        super().__init__()
        self.param_shapes = param_shapes # `param_shapes = [[D_out, D_in], [D_out]]`
        self.cond_embs = nn.Embedding(num_cond_embs, cond_emb_dim)
        self.to_weight = nn.Linear(cond_emb_dim, param_shapes[0][0] * param_shapes[0][1])
        self.to_bias = nn.Linear(cond_emb_dim, param_shapes[1][0])
    
    def forward(self, cond_id):
        if type(cond_id) != list:
            cond_id = [cond_id]

        cond_id = torch.tensor(cond_id).long().to(self.to_weight.weight.device) 
        num_conds = len(cond_id)
        cond_emb = self.cond_embs(cond_id)

        params = []
        for i in range(num_conds):
            predicted_weight = self.to_weight(cond_emb[i])
            predicted_weight = predicted_weight.view((self.param_shapes[0][0], self.param_shapes[0][1]))

            predicted_bias = self.to_bias(cond_emb[i])
            predicted_bias = predicted_bias.view((self.param_shapes[0][1],))

            if num_conds == 1:
                return [predicted_weight, predicted_bias]

            else:
                params.append([predicted_weight, predicted_bias])

        return params


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


class ConvHyperNet(nn.Module):
    def __init__(self, dims, num_cond_embs, cond_emb_dim):
        """
        -----
        Note:
        -----

        If len(dims["num_embed_dims"]) == N, then we need to make
        N optimizers, each of which will contain the parameters of
        `self.weights_decoder[i]` and `self.biases_decoder[i]`.
        One more optimizer is needed to optimize the parameters of
        `self.cond_embs`, `self.fc1`, `self.cn0`.
        """
        super().__init__()
        self.dims = dims
        self.num_cond_embs = num_cond_embs
        self.cond_emb_dim = cond_emb_dim

        self.cond_embs = nn.Embedding(num_cond_embs, cond_emb_dim)
        self.fc1 = nn.Linear(cond_emb_dim, dims["text"])
        self.cn0 = nn.Conv1d(1, dims["intermediate"], 1, 1)
        
        self.weights_decoder = [nn.Conv1d(dims["intermediate"], dims["image"][i], 1, 1) for i in range(dims["num_embed_dims"])]
        self.weights_decoder = nn.ModuleList(self.weights_decoder)

        self.biases_decoder = [nn.Linear(cond_emb_dim, dims["image"][i]) for i in range(dims["num_embed_dims"])]
        self.biases_decoder = nn.ModuleList(self.biases_decoder)

        self.decoder_lookup = {dims["image"][i]: i for i in range(dims["num_embed_dims"])}
        self.act = nn.ReLU()

    def forward(self, cond_id, D_img):
        if type(cond_id) != list:
            cond_id = [cond_id]
        
        cond_id = torch.tensor(cond_id).long().to(self.fc1.weight.device)
        num_conds = len(cond_id)

        params = []
        cond_embs = self.cond_embs(cond_id)
        for i in range(num_conds):
            weight_base = self.fc1(cond_embs[i]) # shape: [cond_emb_dim]
            weight_base = self.act(weight_base).unsqueeze(0).unsqueeze(0) # shape [1, 1, text_embed_dim]
            weight_base = self.cn0(weight_base)

            index = self.decoder_lookup[D_img]
            weight = self.weights_decoder[index](weight_base).squeeze(0)
            bias = self.biases_decoder[index](cond_embs[i])
        
            if num_conds == 1:
                return [weight, bias]
            else:
                params.append([weight, bias])
        
        return params


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
