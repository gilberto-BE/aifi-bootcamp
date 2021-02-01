# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:42:50 2021

@author: Gilberto-BE
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import tune
from ray.tune.schedulers import ASHAScheduler


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


EPOCH_SIZE = 512
TEST_SIZE = 256


def train(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct/total


def train_mnist(config):
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))
         ]
        )

    train_loader = DataLoader(
        datasets.MNIST(
            "~/data",
            train=True,
            download=True,
            transform=mnist_transforms),
        batch_size=64,
        shuffle=True
        )

    test_loader = DataLoader(
        datasets.MNIST(
            "~/data",
            train=False,
            transform=mnist_transforms),
        batch_size=64,
        shuffle=True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"]
        )
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)

        tune.report(mean_accuracy=acc)

        if i % 5 == 0:
            torch.save(model.state_dict(), "./model.pth")

if __name__ == "__main__":
    """Perform async Hyperband optimization ASHA"""

    search_space = {
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "momentum": tune.uniform(0.1, 0.9)
        }

    datasets.MNIST("~/data", train=True, download=True)
    analysis = tune.run(
        train_mnist,
        num_samples=20,
        scheduler=ASHAScheduler(metric='mean_accuracy', mode='max'),
        config=search_space, resources_per_trial={'gpu': 1}
        )

    dfs = analysis.trial_dataframes
    [d.mean_accuracy.plot() for d in dfs.values()]

    """Plot by epoch"""
    ax = None
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Accuracy")

