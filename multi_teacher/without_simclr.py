
import os
import argparse
from typing_extensions import List
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as torch_datasets

from data import MultiTeacherDistillationDataset
from utils import ImageEncoder, MlpMapper


class Trainer():
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

                sim_with_teachers = torch.einsum("bnd,cd->bnc", multiteacher_features, mapped_student_features)
                loss_with_teachers = sum([F.cross_entropy(sim_with_teachers[:, j, :], labels) for j in range(num_teachers)])
                loss_with_teachers = loss_with_teachers / num_teachers

                total_loss = loss_with_teachers
                running_loss += total_loss.item()

                # backward pass
                total_loss.backward()
                optimizer.step()

                bar.update(1)
                bar.set_postfix({"avg_loss": running_loss / (idx+1)})

            running_loss /= len(loader)
            logs["train"][f"epoch_{epoch+1}"] = {"avg_loss": running_loss}

            # saving
            if (epoch+1) in [100]:
                dump = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "logs": logs
                }
                torch.save(dump, os.path.join(self.ckpt_save_folder, f"ckpt_{epoch+1}.pt"))

        return logs


@torch.no_grad()
def evaluate_kmc_cifar10(args, encoder_name, mapper_ckpt):
    image_encoder = ImageEncoder(encoder_name)
    mapper = MlpMapper(args.student_dim, [], args.teacher_dim)
    mapper.load_state_dict(mapper_ckpt)
    mapper = mapper.to(args.device)
    mapper.eval()

    dataset = torch_datasets.CIFAR10(root=args.dataset_root, train=False, download=False, transform=image_encoder.transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    X, y = [], []
    for images, labels in loader:
        images = images.float().to(args.device)
        image_features = mapper(image_encoder.encode_image(images)).cpu()

        X.append(image_features)
        del image_features
        y.append(labels)

    X = torch.cat(X, dim=0).numpy()
    y = torch.cat(y, dim=0).numpy()

    kmc = KMeans(n_clusters=10)
    y_preds = kmc.predict(X, y)
    accuracy = accuracy_score(y, y_preds)
    return round(accuracy * 100, 2)

@torch.no_grad()
def encode_cifar10_train(args):
    teacher_names = ["vit_base_patch16_224", "deit_base_patch16_224", "swin_small_patch4_window7_224.ms_in22k_ft_in1k"]
    teacher_data = {}
    for name in teacher_names:
        encoder = ImageEncoder(name)
        dataset = torch_datasets.CIFAR10(root=args.dataset_root, train=True, download=False, transform=encoder.transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)

        store = []
        bar = tqdm(total=len(loader))
        for images, _ in loader:
            images = images.float().to(args.device)
            image_features = encoder.encode_image(images).cpu()
            store.append(image_features)
            del image_features
            bar.update(1)

        store = torch.cat(store, dim=0)
        teacher_data[name] = store

    torch.save({args.teacher_dim: teacher_data}, os.path.join(args.results_folder, args.experiment_type, args.dataset_name, f"dim_{args.teacher_dim}.pt"))

    student_names = ["visformer_tiny.in1k"]
    student_data = {}
    for name in student_names:
        encoder = ImageEncoder(name)
        dataset = torch_datasets.CIFAR10(root=args.dataset_root, train=True, download=False, transform=encoder.transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        store = []
        bar = tqdm(total=len(loader))
        for images, _ in loader:
            images = images.float().to(args.device)
            image_features = encoder.encode_image(images).cpu()
            store.append(image_features)
            bar.update(1)

        store = torch.cat(store, dim=0)
        student_data[name] = store

    torch.save({args.student_dim: student_data}, os.path.join(args.results_folder, args.experiment_type, args.dataset_name, f"dim_{args.student_dim}.pt"))


def main(args):
    dataset = MultiTeacherDistillationDataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    model = MlpMapper(args.student_dim, [], args.teacher_dim)

    initial_kmc_accuracy = evaluate_kmc_cifar10(args, dataset.student_model_name, model.state_dict())
    print("K-Means accuracy on CIFAR-10 for student model before distillation:")
    print(initial_kmc_accuracy)
    print(" ")

    trainer = Trainer(args)
    train_logs = trainer.train(model, loader)
    final_kmc_accuracy = evaluate_kmc_cifar10(args, dataset.student_model_name, model.state_dict())
    print("K-Means accuracy on CIFAR-10 for student model after multi-teacher distillation:")
    print(final_kmc_accuracy)
    print(" ")


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scheduler", type=str, default="off")
    parser.add_argument("--self-loss", type=str, default="on")
    parser.add_argument("--multi-teacher-loss", type=str, default="on")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-teachers", type=int, default=3)

    parser.add_argument("--experiment-type", type=str, default="multi_distil")
    parser.add_argument("--experiment-name", type=str, default="test_0")
    parser.add_argument("--checkpoint-folder", type=str, default="../checkpoints")
    parser.add_argument("--dataset-name", type=str, default="cifar10_train")
    parser.add_argument("--logs-folder", type=str, default="../logs")
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--dataset-root", type=str, default="/home/mila/s/sparsha.mishra/scratch/cifar10_torchvision")

    parser.add_argument("--student-index", type=int, default=0)
    parser.add_argument("--teacher-indices", type=str, default="0,1,2")
    parser.add_argument("--student-dim", type=int, default=384)
    parser.add_argument("--teacher-dim", type=int, default=768)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args()
    main(args)
