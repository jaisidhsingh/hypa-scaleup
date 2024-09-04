import torch


def make_dummy_dataset():
    data = {}
    data["vit_base"] = torch.randn(100, 768)
    data["deit_base"] = torch.randn(100, 768)
    data["swin_small"] = torch.randn(100, 768)
    data["convnext_small"] = torch.randn(100, 768)

    data2 = {}
    data2["vit_small"] = torch.randn(100, 384)
    data2["deit_small"] = torch.randn(100, 384)
    data2["vit_tiny"] = torch.randn(100, 384)
    data2["deit_tiny"] = torch.randn(100, 384)

    data3 = {}
    data3["vit_large"] = torch.randn(100, 1024)
    data3["deit_large"] = torch.randn(100, 1024)
    data3["resnet_50"] = torch.randn(100, 1024)
    data3["swin_base"] = torch.randn(100, 1024)

    results = {
        384: data2,
        768: data,
        1024: data3
    }
    torch.save(results, "datasets/random_image_embeddings.pt")

    data = {768: {"sentence-t5-base": torch.randn(100, 768)}}
    torch.save(data, "datasets/random_text_embeddings.pt")


def test_generators():
    def count_up_to(n):
        count = 1
        while count <= n:
            yield count
            count += 1

    loader = count_up_to(5)
    print(next(loader))
    print(next(loader))
    print(next(loader))


if __name__ == "__main__":
    test_generators()
