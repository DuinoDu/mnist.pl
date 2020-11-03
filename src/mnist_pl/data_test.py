import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mnist_pl import data
from torchvision import transforms


def test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = data.Dataset('', train=True, transform=transform)
    
    for item in dataset:
        x, y = item
        break


if __name__ == "__main__":
    test_dataset()
