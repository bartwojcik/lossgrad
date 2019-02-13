import argparse
from math import ceil
from pathlib import Path

import torch
import torchvision
from torch import nn

from lossgrad import LossgradOptimizer
from utils import setup_dev, create_loader, plot_results, test_autoencoder, plot_examples

X_IMAGE_SIZE = 28
Y_IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
IMAGE_SIZE = X_IMAGE_SIZE * Y_IMAGE_SIZE * IMAGE_CHANNELS


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(IMAGE_SIZE, 200), nn.ReLU(True),
            nn.Linear(200, 100), nn.ReLU(True),
            nn.Linear(100, 50), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 100), nn.ReLU(True),
            nn.Linear(100, 200), nn.ReLU(True),
            nn.Linear(200, IMAGE_SIZE), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, IMAGE_CHANNELS, Y_IMAGE_SIZE, X_IMAGE_SIZE)
        return x


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
        torch.nn.init.constant_(m.bias, 0.2)


def train_model(device, model, criterion, train_data, test_data, batch_size, epochs, optimizer,
                scheduler=None, num_points=200, test_batches=20):
    batches_per_epoch = ceil(len(train_data) / batch_size)
    max_batch = max_epoch * batches_per_epoch
    xs = set(round(x.item()) for x in torch.linspace(0, max_batch - 1, steps=num_points, device='cpu'))

    results = {'x': [], 'train_losses': [], 'test_losses': [], 'train_data_len': len(train_data),
               'batch_size': batch_size}

    model.train()
    train_loader = create_loader(train_data, batch_size)
    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()
        for batch, (X, _) in enumerate(train_loader):
            X = X.to(device)

            X_pred = model(X)
            loss = criterion(X_pred, X)
            model.zero_grad()
            loss.backward()
            if isinstance(optimizer, LossgradOptimizer):
                optimizer.step(X, X, loss)
            else:
                optimizer.step()

            x = epoch * batches_per_epoch + batch
            if x in xs:
                train_loss = test_autoencoder(device, model, train_data, criterion, batch_size,
                                              batches=test_batches)
                test_loss = test_autoencoder(device, model, test_data, criterion, batch_size, batches=test_batches)
                results['x'].append(x)
                results['train_losses'].append(train_loss)
                results['test_losses'].append(test_loss)
                print('Epoch: {}, Batch: {}, Train Loss: {:.3f}, Test Loss: {:.3f}'.format(epoch, batch, train_loss,
                                                                                           test_loss))
    results['model'] = model.state_dict()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='Directory containing the datasets.', type=Path,
                        default=Path.home() / '.datasets')
    args = parser.parse_args()

    device = setup_dev()

    dataset = torchvision.datasets.FashionMNIST
    train_data = dataset(root=str(args.dataset_dir), train=True,
                         transform=torchvision.transforms.ToTensor(), download=True)
    test_data = dataset(root=str(args.dataset_dir), train=False,
                        transform=torchvision.transforms.ToTensor(), download=True)

    max_epoch = 30
    batch_size = 100

    criterion = nn.MSELoss()

    net = Net()
    runs = {}

    net.apply(init_weights)
    optimizer = LossgradOptimizer(torch.optim.SGD(net.parameters(), lr=1e-4), net, criterion)
    runs['lossgrad'] = train_model(device, net, criterion, train_data, test_data, batch_size, max_epoch, optimizer)
    train_loss = test_autoencoder(device, net, train_data, criterion, batch_size)
    test_loss = test_autoencoder(device, net, test_data, criterion, batch_size)
    print(f'Final train loss: {train_loss}')
    print(f'Final test loss: {test_loss}')

    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
    runs['sgd'] = train_model(device, net, criterion, train_data, test_data, batch_size, max_epoch, optimizer)
    train_loss = test_autoencoder(device, net, train_data, criterion, batch_size)
    test_loss = test_autoencoder(device, net, test_data, criterion, batch_size)
    print(f'Final train loss: {train_loss}')
    print(f'Final test loss: {test_loss}')

    plots_dir_name = 'plots'
    plot_results(plots_dir_name, runs, 'train_losses')
    plot_results(plots_dir_name, runs, 'test_losses')
    plot_examples(plots_dir_name, device, net, test_data, runs)
