import os
import argparse
from typing_extensions import List
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as torch_datasets

from data import MultiTeacherDistillationDataset


class MultiTeacherDistillator():
    def __init__(self, args):
        self.device = args.device
        self.ckpt_save_folder = os.path.join(args.checkpoint_folder, args.experiment_type, args.experiment_name)
        os.makedirs(self.ckpt_save_folder, exist_ok=True)

        self.logs_save_folder = os.path.join(args.logs_folder, args.experiment_type, args.experiment_name)
        os.makedirs(self.logs_save_folder, exist_ok=True)
        self.args = args

    def train(self, model: nn.Module, loader: DataLoader):
        num_teachers = self.args.num_teachers
        num_epochs = self.args.num_epochs
        bar = tqdm(total=num_epochs)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        scheduler = None

        logs = {"train": {}}
        for epoch in range(num_epochs):
            logs["train"][f"epoch_{epoch+1}"] = {}

            correct, total = 0, 0
            running_loss = 0

            for idx, (student_features, multiteacher_features) in enumerate(loader):
                batch_size = student_features.shape[0]
                student_features = student_features.float().to(self.device)

                multiteacher_features = multiteacher_features.float().to(self.device)
                labels = torch.arange(batch_size, dtype=torch.long).to(self.device)

                if scheduler is not None:
                    step = epoch * len(loader) + idx
                    scheduler(step)

                # zero grads
                optimizer.zero_grad()

                # forward pass
                mapped_student_features = model(student_features)
                mapped_student_features = mapped_student_features.norm(dim=-1, keepdim=True)

                # compute loss
                sim_with_teachers = multiteacher_features @ mapped_student_features.T
                loss_with_teachers = sum([F.cross_entropy(sim_with_teachers[:, j, :], labels) for j in range(num_teachers)])

                loss_with_self = 0
                if self.args.self_loss == "on":
                    sim_with_self = mapped_student_features @ mapped_student_features.T
                    loss_with_self = F.cross_entropy(sim_with_self, labels)

                total_loss = loss_with_teachers + loss_with_self
                running_loss += total_loss.item()

                # backward pass
                total_loss.backward()
                optimizer.step()

            running_loss /= len(loader)
            logs["train"][f"epoch_{epoch+1}"] = {"avg_loss": running_loss}

            # saving
            if (epoch+1) in [1, 5, 10, 20, 40]:
                dump = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "logs": logs
                }
                torch.save(dump, os.path.join(self.ckpt_save_folder, f"ckpt_{epoch+1}.pt"))

        return logs


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=int(2**12))
    parser.add_argument("--scheduler", type=str, default="off")
    parser.add_argument("--self-loss", type=str, default="on")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-teachers", type=int, default=5)

    parser.add_argument("--experiment-type", type=str, default="multi_distil")
    parser.add_argument("--experiment-name", type=str, default="test_0")
    parser.add_argument("--checkpoint-folder", type=str, default="../checkpoints")
    parser.add_argument("--logs-folder", type=str, default="../logs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args()
    dataset = MultiTeacherDistillationDataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    model = nn.Linear(384, 768)
    trainer = MultiTeacherDistillator(args)
    trainer.train(model, loader)
