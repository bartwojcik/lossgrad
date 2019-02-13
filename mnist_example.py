import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.transforms import transforms

from lossgrad import LossgradOptimizer
from utils import create_loader, setup_dev, test_classification

IMAGE_SIZE = 28 * 28
CLASSES = 10


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = F.relu(self.fc1(x), True)
        x = F.relu(self.fc2(x), True)
        x = self.fc3(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='Directory containing the datasets.', type=Path,
                        default=Path.home() / '.datasets')
    args = parser.parse_args()

    device = setup_dev()

    dataset = torchvision.datasets.MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_data = dataset(root=str(args.dataset_dir), train=True,
                         transform=transform, download=True)
    test_data = dataset(root=str(args.dataset_dir), train=False,
                        transform=transform, download=True)

    max_epoch = 30
    batch_size = 100

    criterion = nn.CrossEntropyLoss()

    net = Net()
    net.apply(init_weights)

    optimizer = LossgradOptimizer(torch.optim.SGD(net.parameters(), lr=1e-4), net, criterion)

    net.train()
    train_loader = create_loader(train_data, batch_size)
    for epoch in range(max_epoch):
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            y_pred = net(X)
            loss = criterion(y_pred, y)
            net.zero_grad()
            loss.backward()
            optimizer.step(X, y, loss)

            if i % 100 == 0:
                print(f'Epoch: {epoch} Batch: {i} Train loss: {loss:.6f}')

    train_loss, train_acc = test_classification(device, net, train_data, criterion, batch_size * 10)
    test_loss, test_acc = test_classification(device, net, test_data, criterion, batch_size * 10)

    print(f'Final train loss: {train_loss} Final train accuracy: {train_acc}')
    print(f'Test loss: {test_loss} Test accuracy: {test_acc}')
