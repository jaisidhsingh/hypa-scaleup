from torchvision.transforms import transforms
from torchvision import transforms, datasets
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import os
np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            print("Dataset not found.")
        else:
            return dataset_fn()


class MultiTeacherDistillationDataset(Dataset):
    def __init__(self, args):
        self.teacher_features = torch.load(os.path.join(
            args.results_folder, args.experiment_type, args.dataset_name, f"dim_{args.teacher_dim}.pt"
        ))[args.teacher_dim]

        total_teacher_keys = list(self.teacher_features.keys())
        unwanted_keys = [total_teacher_keys[j] for j in range(len(total_teacher_keys)) if str(j) not in args.teacher_indices.split(",")]
        for k in unwanted_keys:
            self.teacher_features.pop(k)

        self.teacher_model_names = [k for k in self.teacher_features.keys()]

        self.student_features = torch.load(os.path.join(
            args.results_folder, args.experiment_type, args.dataset_name, f"dim_{args.student_dim}.pt"
        ))[args.student_dim]

        total_student_keys = list(self.student_features.keys())
        unwanted_keys = [total_student_keys[j] for j in range(len(total_student_keys)) if j != args.student_index]
        for k in unwanted_keys:
            self.student_features.pop(k)

        self.student_model_name = [k for k in self.student_features.keys()][0]
        self.num_samples = self.student_features[list(self.student_features.keys())[0]].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        teacher_features = torch.cat(
            [self.teacher_features[k][idx].unsqueeze(0) for k in self.teacher_features.keys()],
            dim=0
        )
        student_key = list(self.student_features.keys())[0]
        student_features = self.student_features[student_key][idx]
        return student_features, teacher_features
