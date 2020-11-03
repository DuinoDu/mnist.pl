import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mnist_pl.model import Model
import torch


def test_model():
    model = Model(hidden_dim=128, learning_rate=0.001)
    x = torch.zeros((1, 1, 28, 28))
    model = model
    out = model(x)


if __name__ == "__main__":
    test_model()
