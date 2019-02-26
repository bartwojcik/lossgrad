# LOSSGRAD
This is an implementation of the LOSSGRAD optimization algorithm for PyTorch.
The algorithm has been implemented in the paper:
[LOSSGRAD: automatic learning rate in gradient descent](https://arxiv.org/abs/1902.07656)


### Usage
Import the `lossgrad` module and simply wrap your SGD optimizer with `LossgradOptimizer` instance, e.g.:
```python
optimizer = LossgradOptimizer(torch.optim.SGD(net.parameters(), lr=1e-4), net, criterion)
```
and pass the required additional arguments to the optimizer's `step` method:
```python
optimizer.step(X, y, loss)
```

### Examples
A simple example for MNIST dataset is shown in
[mnist_example.py](https://github.com/bartwojcik/lossgrad/blob/master/mnist_example.py).

A more complicated example for an autoencoder trained on Fashion-MNIST (with figures and image samples) is shown in
[autoencoder_fmnist_example.py](https://github.com/bartwojcik/lossgrad/blob/master/autoencoder_fmnist_example.py).


### Citing
If you use the algorithm you can cite our paper:
```
@article{wojcik2019lossgrad,
  title={LOSSGRAD: automatic learning rate in gradient descent},
  author={W{\'o}jcik, Bartosz and Maziarka, {\L}ukasz and Tabor, Jacek},
  journal={arXiv preprint arXiv:1902.07656},
  year={2019}
}
```