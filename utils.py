import traceback
from itertools import cycle
from pathlib import Path

import seaborn
import torch
from matplotlib import pyplot as plt

seaborn.set()
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def setup_dev():
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        device = 'cpu'
    return device


def create_loader(data, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(dataset=data, batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=torch.cuda.is_available(),
                                       num_workers=num_workers)


def test_classification(device, model, data, criterion, batch_size, batches=0):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct, total = 0, 0
        for batch, (X, y) in enumerate(create_loader(data, batch_size)):
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            y_pred_max = y_pred.argmax(dim=1)
            correct += (y_pred_max == y).sum().item()
            total += y.size(0)
            if batch > batches > 0:
                break
    model.train()
    # loss, acc
    return running_loss / batch, correct / total


def test_autoencoder(device, model, data, criterion, batch_size, batches=0):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch, (X, _) in enumerate(create_loader(data, batch_size)):
            X = X.to(device)
            X_pred = model(X)
            loss = criterion(X_pred, X)
            running_loss += loss.item()
            if batch > batches > 0:
                break
    model.train()
    return running_loss / batch


def to_epoch(x, batches_per_epoch):
    return x / batches_per_epoch


def plot_results(plots_dir, runs, result_name):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    imgpath = plots_dir / f'{result_name}.png'
    if not imgpath.is_file():
        try:
            plt.figure(figsize=(16, 9))
            c_cycler = cycle(seaborn.color_palette())
            for name, results in runs.items():
                c = next(c_cycler)
                if result_name in results and len(results[result_name]):
                    train_batches = results['train_data_len'] // results['batch_size']
                    plt.plot([to_epoch(x, train_batches) for x in
                              range(len(results[result_name]))],
                             results[result_name],
                             '.-', label=name, color=c)
            plt.xlabel('Epoch')
            plt.title(result_name)
            plt.legend()
            plt.savefig(imgpath)
        except Exception:
            traceback.print_exc()


def imsave(img, path):
    npimg = img.squeeze().cpu().numpy()
    plt.imsave(path, npimg)


def plot_examples(plot_dir, device, model, data, runs, images=5):
    plots_dir = Path(plot_dir)
    for name, results in runs.items():
        model.load_state_dict(results['model'])
        for X, _ in create_loader(data, images):
            X = X.to(device)
            X_pred = model(X).detach()
            X = X.detach()
            sample_dir = plots_dir / f'{name}_samples'
            sample_dir.mkdir(parents=True, exist_ok=True)
            for i in range(images):
                imsave(X_pred[i], str(sample_dir / f'{i}.png'))
                imsave(X[i], str(sample_dir / f'{i}_input.png'))
            break
